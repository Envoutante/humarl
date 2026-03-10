from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import os
import numpy as np
import torch as th
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
        self.train_pred_returns = []
        self.test_pred_returns = []
        self.train_pred_return_abs_errors = []
        self.test_pred_return_abs_errors = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        """
         * @author hyr
         * @modified 2026-03-06-15:09
         * @description 引入奖励网络, 以实时计算奖励预测值
        """
        self.reward_mixer = None
        self.reward_predict_enabled = False
        self.pred_return_warmup_steps = int(
            getattr(self.args, "pred_return_warmup_steps", 0)
        )
        if getattr(self.args, "reward_checkpoint_path", ""):
            self.pred_return_start_t = 0
        else:
            self.pred_return_start_t = int(
                getattr(self.args, "q_tot_stage_steps", 0)
                + self.pred_return_warmup_steps
            )

        """
         * @author hyr
         * @modified 2026-01-05-15:49
         * @description 初始化 episode return 日志文件（仅在需要记录时初始化）
        """
        self.episode_return_log_path = None
        if self.args.log_episode_return:
            episode_return_log_dir = os.path.join(
                os.path.abspath(self.args.local_results_path), "episode_return_logs"
            )
            os.makedirs(episode_return_log_dir, exist_ok=True)
            unique_token = getattr(self.args, "unique_token", "default")
            self.episode_return_log_path = os.path.join(
                episode_return_log_dir, f"{unique_token}.csv"
            )
            if not os.path.exists(self.episode_return_log_path):
                with open(self.episode_return_log_path, "w", encoding="utf-8") as f:
                    f.write("episode_idx, episode_return\n")

        """
         * @author hyr
         * @modified 2026-01-05-15:49
         * @description 初始化 TransitionStorage, 用于存储训练时的 transitions
        """
        self.args.env_info = self.get_env_info()
        if self.args.collect_transitions:
            storage_dir = os.path.join(
                os.path.abspath(self.args.local_results_path), "collected_transitions"
            )
            os.makedirs(storage_dir, exist_ok=True)
            unique_token = self.args.unique_token
            file_path = os.path.join(storage_dir, f"transitions_{unique_token}.h5")
            self.transition_storage = TransitionStorage(self.args, file_path)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

        if self.args.reward_mixer:
            # Use learner's reward_mixer by reference to keep prediction logging
            # consistent with the reward network actually used during Q training.
            self.reward_predict_enabled = False

    def set_reward_mixer(self, reward_mixer):
        """Bind runner to learner reward mixer so logging uses the same weights."""
        self.reward_mixer = reward_mixer
        self.reward_predict_enabled = reward_mixer is not None

    def _pred_return_stats_enabled(self):
        if not self.reward_predict_enabled:
            return False
        return self.t_env >= self.pred_return_start_t

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

        while not terminated:

            # 收集状态转移前的数据并存入 self.batch
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # mac 根据 self.batch 选择动作
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            # 在环境中执行动作
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward  # 更新当前回合的总奖励 episode_return

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
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # mac 根据 self.batch 选择动作并存入 self.batch
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        """
         * @author hyr
         * @modified 2026-03-09-15:24
         * @description 记录 reward 网络的预测值
        """
        episode_return_pred = None
        if self.reward_predict_enabled and self.reward_mixer is not None:
            with th.no_grad():
                prev_training = self.reward_mixer.training
                self.reward_mixer.eval()
                _, global_reward_pred = self.reward_mixer(self.batch)
                # Keep shapes aligned as [batch, T, 1] to avoid unintended broadcasting.
                reward_mask = self.batch["filled"][:, :-1].float()
                return_pred_batch = (global_reward_pred * reward_mask).sum(dim=1)
                episode_return_pred = return_pred_batch.squeeze(-1).item()
                if prev_training:
                    self.reward_mixer.train()

        # 更新总时间步 self.t_env
        if not test_mode:
            self.t_env += self.t

        # 记录日志
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_pred_returns = (
            self.test_pred_returns if test_mode else self.train_pred_returns
        )
        cur_pred_return_abs_errors = (
            self.test_pred_return_abs_errors
            if test_mode
            else self.train_pred_return_abs_errors
        )
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)
        if episode_return_pred is not None and self._pred_return_stats_enabled():
            cur_pred_returns.append(episode_return_pred)
            cur_pred_return_abs_errors.append(
                abs(float(episode_return_pred) - float(episode_return))
            )

        """
         * @author hyr
         * @modified 2026-01-05-15:49
         * @description 记录每个 episode 的 reward 之和到 CSV 文件（仅在 test_mode 且需要记录时）
        """
        if (
            test_mode
            and self.args.log_episode_return
            and self.episode_return_log_path is not None
        ):
            episode_idx = len(cur_returns) - 1  # 当前 episode 的序号（从0开始）
            with open(self.episode_return_log_path, "a", encoding="utf-8") as f:
                f.write(f"{episode_idx}, {episode_return}\n")

        """
         * @author hyr
         * @modified 2026-01-05-15:49
         * @description 保存训练时的 transitions 用于离线强化学习
        """
        if (
            not test_mode
            and self.args.collect_transitions
            and self.transition_storage is not None
        ):
            self.transition_storage.save_transition_batch(self.batch)

        # 记录日志
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(
                cur_returns,
                cur_pred_returns,
                cur_pred_return_abs_errors,
                cur_stats,
                log_prefix,
            )
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(
                cur_returns,
                cur_pred_returns,
                cur_pred_return_abs_errors,
                cur_stats,
                log_prefix,
            )
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, pred_returns, pred_abs_errors, stats, prefix):
        """
        * @author hyr
        * @modified 2026-03-06-10:32
        * @description 修改日志字段名称
        """
        true_return_mean = np.mean(returns)
        true_return_std = np.std(returns)
        if len(pred_returns) > 0:
            pred_return_mean = np.mean(pred_returns)
            pred_return_std = np.std(pred_returns)
            pred_return_mae = np.mean(pred_abs_errors) if len(pred_abs_errors) > 0 else 0.0
            self.logger.log_stat_group(
                prefix + "return_mean",
                {"true": true_return_mean, "pred": pred_return_mean},
                self.t_env,
            )
            self.logger.log_stat_group(
                prefix + "return_std",
                {"true": true_return_std, "pred": pred_return_std},
                self.t_env,
            )
            self.logger.log_stat(prefix + "pred_return_mae", pred_return_mae, self.t_env)
            self.logger.log_stat(
                prefix + "pred_return_valid_count", len(pred_returns), self.t_env
            )
        else:
            self.logger.log_stat(prefix + "return_mean", true_return_mean, self.t_env)
            self.logger.log_stat(prefix + "return_std", true_return_std, self.t_env)
            self.logger.log_stat(prefix + "pred_return_valid_count", 0, self.t_env)

        returns.clear()
        pred_returns.clear()
        pred_abs_errors.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
