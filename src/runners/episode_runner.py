from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import os
import numpy as np
from utils.transition_storage import TransitionStorage


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        """
        新增
        测试阶段 reward 结构化日志文件（按实验 token 区分）
        """
        if self.args.save_reward:
            rewards_log_dir = os.path.join(os.path.abspath(self.args.local_results_path), "reward_logs")
            os.makedirs(rewards_log_dir, exist_ok=True)
            unique_token = getattr(self.args, "unique_token", "default")
            self.rewards_log_path = os.path.join(rewards_log_dir, f"rewards_{unique_token}.csv")
            if not os.path.exists(self.rewards_log_path):
                with open(self.rewards_log_path, "w", encoding="utf-8") as f:
                    f.write("episode_idx, reward_0, reward_1, ...\n")

        """
        新增
        离线强化学习数据收集：存储训练时的 transitions
        """
        self.args.env_info = self.get_env_info()
        if self.args.collect_transitions:
            storage_dir = os.path.join(os.path.abspath(self.args.local_results_path), "collected_transitions")
            os.makedirs(storage_dir, exist_ok=True)
            unique_token = self.args.unique_token
            file_path = os.path.join(storage_dir, f"transitions_{unique_token}.h5")
            self.transition_storage = TransitionStorage(self.args, file_path)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        # 初始化环境、terminated、episode_return 和 mac 的 hidden state
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        """
        新增
        记录每个 episode 的所有 reward
        """
        episode_rewards = [] if test_mode and self.args.save_reward else None

        while not terminated:

            # 收集状态转移前的数据并存入 self.batch
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # mac 根据 self.batch 选择动作
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # 在环境中执行动作
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward  # 更新当前回合的总奖励 episode_return

            """
            新增
            记录每个时间步的 reward
            """
            if test_mode and self.args.save_reward:
                episode_rewards.append(reward)

            # 收集状态转移后的数据并存入 self.batch
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            # 更新当前 episode 的时间步 self.t
            self.t += 1

        # 收集“最后一个时间步”状态转移前的数据并存入 self.batch
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # mac 根据 self.batch 选择动作并存入 self.batch
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # 更新总时间步 self.t_env
        if not test_mode:
            self.t_env += self.t

        # 记录日志
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)

        """
        新增
        把记录的 reward 写入 csv 文件
        """
        if test_mode and self.args.save_reward and episode_rewards:
            episode_idx = len(self.test_returns)
            row = ", ".join([str(episode_idx)] + [str(reward) for reward in episode_rewards])
            with open(self.rewards_log_path, "a", encoding="utf-8") as rewards_file:
                rewards_file.write(f"{row}\n")

        """
        新增
        保存训练时的 transitions 用于离线强化学习
        """
        if not test_mode and self.args.collect_transitions and self.transition_storage is not None:
            self.transition_storage.save_transition_batch(self.batch)

        # 记录日志
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
