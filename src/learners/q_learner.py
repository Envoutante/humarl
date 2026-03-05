import copy
import os
import numpy as np
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.reward_mixer import RewardMixer
from utils.transition_storage import TransitionStorage
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        """
        新增 reward_mixer
        """
        self.reward_mixer = RewardMixer(args)
        self.reward_optimizer = RMSprop(
            params=self.reward_mixer.parameters(),
            lr=args.lr,
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )
        """
        新增 target_reward_mixer
        """
        self.target_reward_mixer = copy.deepcopy(self.reward_mixer)
        """
        新增
        """
        self.reward_log_t = -self.args.learner_log_interval - 1
        self.reward_save_t = 0

        """
         * @author hyr
         * @modified 2025-11-25-17:30
         * @description 记录 individual_rewards
        """
        self.individual_rewards_log_path = None
        if self.args.reward_mixer and self.args.log_individual_reward:
            logs_dir = os.path.join(
                os.path.abspath(self.args.local_results_path), "individual_reward_logs"
            )
            os.makedirs(logs_dir, exist_ok=True)
            unique_token = getattr(self.args, "unique_token", "default")
            self.individual_rewards_log_path = os.path.join(
                logs_dir, f"{unique_token}.csv"
            )
            if not os.path.exists(self.individual_rewards_log_path):
                with open(self.individual_rewards_log_path, "w", encoding="utf-8") as f:
                    header_cells = [
                        "episode_idx",
                        "total_reward",
                        "total_pred_reward",
                    ] + [
                        f"step_{i}" for i in range(self.args.env_info["episode_limit"])
                    ]
                    f.write(", ".join(header_cells) + "\n")

        """
         * @author hyr
         * @modified 2025-11-25-17:00
         * @description 避免反复实例化 TransitionStorage
        """
        if self.args.use_transitions:
            # 根据 file_path 实例化 transition_storage
            storage_dir = os.path.join(
                os.path.abspath(self.args.local_results_path), "collected_transitions"
            )
            file_path = os.path.join(
                storage_dir, self.args.transitions_filename + ".h5"
            )
            self.transition_storage = TransitionStorage(self.args, file_path)

    """
    训练 reward 网络
    """

    def train_reward_network(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if not self.args.reward_mixer:
            return None, None

        """
        新增
        从 transition_storage 中获取一个 batch
        """
        if self.args.use_transitions:
            # 从 H5 文件中加载 transition
            batch = self.transition_storage.load_transition_batch(
                batch_size=self.args.reward_batch_size
            )
            # 移动数据到指定的 device
            batch = {
                key: th.from_numpy(value).to(self.args.device)
                for key, value in batch.items()
            }
            # 截断数据为当前 batch 中最长 episode 的长度
            max_ep_t = th.sum(batch["filled"], 1).max(0)[0]
            batch = {key: tensor[:, :max_ep_t] for key, tensor in batch.items()}
            # 更新 t_env
            total_steps = batch["filled"].sum().item()
            t_env += total_steps

        # 从 batch 中取出相关数据
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # 前向传播
        _, global_reward_pred = self.reward_mixer(batch)

        # 计算损失
        true_global_rewards = batch["reward"][:, :-1]
        reward_error = global_reward_pred - true_global_rewards
        reward_mask = mask.expand_as(reward_error)
        masked_reward_error = reward_error * reward_mask
        training_reward_loss = (masked_reward_error**2).sum() / reward_mask.sum()

        # 反向传播
        self.reward_optimizer.zero_grad()
        training_reward_loss.backward()
        self.reward_optimizer.step()

        # 记录日志
        if t_env - self.reward_log_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "reward_loss/train", training_reward_loss.item(), t_env
            )
            self.reward_log_t = t_env

        return t_env

    """
    训练 Q 网络
    """

    def train_q_network(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 从 batch 中取出相关数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 遍历所有时间步以计算每个动作的 Q 值并放入 mac_out
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 在时间步维度上拼接

        # 以 actions 为索引保留对应的 Q 值，并移除最后一个维度
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )

        # 计算目标 Q 值
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # 由于目标 Q 值是 Q_{t+1} 因此不需要第一个 Q 值
        target_mac_out = th.stack(
            target_mac_out[1:], dim=1
        )  # [batch_size, seq_len-1, n_agents, n_actions]

        # 去除掉不可用的动作
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # 选择目标网络的最大 Q 值 (贝尔曼最优公式)
        if self.args.double_q:  # 使用 Double DQN
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(
                3
            )  # [batch_size, seq_len-1, n_agents]
        else:  # 不使用 Double DQN
            target_max_qvals = target_mac_out.max(dim=3)[
                0
            ]  # [batch_size, seq_len-1, n_agents]

        """
         * @author hyr
         * @modified 2025-11-26-15:55
         * @description 新增的优化目标
        """
        if self.args.reward_mixer:
            # 计算个体 reward
            with th.no_grad():
                individual_rewards, global_reward_pred = self.reward_mixer(batch)

            """
             * @author hyr
             * @modified 2026-01-04-10:19
             * @description 记录奖励网络在测试阶段中的损失
            """
            true_global_rewards = batch["reward"][:, :-1]
            reward_error = global_reward_pred - true_global_rewards
            reward_mask = mask.expand_as(reward_error)
            masked_reward_error = reward_error * reward_mask
            testing_reward_loss = (masked_reward_error**2).sum() / reward_mask.sum()

            # 对个体 reward 进行掩码
            masked_individual_rewards = individual_rewards.squeeze(-1) * mask
            individual_rewards = (
                masked_individual_rewards.detach()
            )  # [batch_size, seq_len-1, n_agents]

            # 混合 reward
            mixd_reward = (
                self.args.reward_weight * rewards
                + (1 - self.args.reward_weight) * individual_rewards
            )
            targets = (
                mixd_reward + self.args.gamma * (1 - terminated) * target_max_qvals
            )

            # 记录 rewards 和 individual_rewards
            self._log_reward(
                rewards,
                individual_rewards,
                global_reward_pred,
                mask,
                t_env,
                episode_num,
            )

            # 计算 TD-error
            td_error = chosen_action_qvals - targets.detach()

            q_mask = mask.expand_as(td_error)

            # mask 掉无效的 TD-error
            masked_td_error = td_error * q_mask

            # 计算 L2 损失 (仅对实际数据取平均)
            q_loss_2 = (masked_td_error**2).sum() / q_mask.sum()

        """
         * @author hyr
         * @modified 2025-11-26-16:12
         * @description 原始的优化目标
        """
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # 计算 TD-error
        td_error = chosen_action_qvals - targets.detach()

        q_mask = mask.expand_as(td_error)

        # mask 掉无效的 TD-error
        masked_td_error = td_error * q_mask

        # 计算 L2 损失 (仅对实际数据取平均)
        q_loss_1 = (masked_td_error**2).sum() / q_mask.sum()

        """
         * @author hyr
         * @modified 2025-11-28-17:55
         * @description 汇总 loss
        """
        if self.args.reward_mixer:
            q_loss = (
                self.args.loss_weight * q_loss_1
                + (1 - self.args.loss_weight) * q_loss_2
            )
        else:
            q_loss = q_loss_1

        # 更新参数
        self.optimiser.zero_grad()
        q_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 更新目标网络
        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 记录日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            if self.args.reward_mixer:
                self.logger.log_stat("q_loss_2", q_loss_2.item(), t_env)
                self.logger.log_stat(
                    "reward_loss/test", testing_reward_loss.item(), t_env
                )
            self.logger.log_stat("q_loss_1", q_loss_1.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = q_mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * q_mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * q_mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _log_reward(
        self,
        rewards,
        individual_rewards,
        global_reward_pred,
        mask,
        t_env: int,
        episode_num: int,
    ):
        """
        * @author hyr
        * @modified 2025-11-25-17:40
        * @description 记录 rewards, global_reward_pred 和 individual_rewards
        """
        if self.individual_rewards_log_path is None:
            return

        # 修复：这里应该是 reward_save_t 而不是 rewards_log_stats_t
        if (
            t_env - self.reward_save_t >= self.args.reward_save_interval
            or self.reward_save_t == 0
        ):
            batch_size, seq_len, _ = individual_rewards.shape
            for episode in range(batch_size):
                step_cells = []
                # 计算当前 episode 在所有有效时间步上的 reward 之和
                mask_ep = mask[episode, :seq_len]
                if th.is_tensor(mask_ep):
                    mask_ep_np = mask_ep.cpu().numpy()
                else:
                    mask_ep_np = np.array(mask_ep)
                # squeeze 奖励到标量，按 mask 过滤
                episode_reward_ep = rewards[episode, :seq_len]
                if th.is_tensor(episode_reward_ep):
                    episode_reward_ep = episode_reward_ep.cpu().numpy()
                # 确保 rewards 是 1D 数组，如果形状是 (seq_len, 1) 则 squeeze
                episode_reward_ep = episode_reward_ep.squeeze()
                # 计算预测的总奖励
                episode_pred_reward_ep = global_reward_pred[episode, :seq_len]
                if th.is_tensor(episode_pred_reward_ep):
                    episode_pred_reward_ep = episode_pred_reward_ep.cpu().numpy()
                # 确保预测奖励是 1D 数组
                episode_pred_reward_ep = episode_pred_reward_ep.squeeze()
                # 确保 mask 也是 1D 数组
                mask_ep_np = mask_ep_np.squeeze()
                # 只对有效的时间步求和（mask > 0）
                episode_total_reward = float(np.sum(episode_reward_ep * mask_ep_np))
                episode_total_pred_reward = float(
                    np.sum(episode_pred_reward_ep * mask_ep_np)
                )
                for t_idx in range(seq_len):
                    # 若该时间步被 mask（填充），则不写入内容
                    if mask_ep_np[t_idx] <= 0:
                        step_cells.append("")
                        continue
                    total_reward = rewards[episode, t_idx].item()
                    pred_reward = global_reward_pred[episode, t_idx].item()
                    # 修改：使用分号 ; 代替逗号，避免破坏 CSV 列结构
                    # 或者去掉 .tolist() 手动拼接，控制分隔符
                    step_rewards_str = ";".join(
                        map(str, individual_rewards[episode, t_idx].tolist())
                    )
                    # 格式：真实的总 reward - 预测的总 reward = [子 reward; 子 reward; ...]
                    cell = f"{total_reward}&{pred_reward}=[{step_rewards_str}]"
                    step_cells.append(cell)

                # episode_num 已经包含当前 batch 的 episode，需要减去 batch_size 来计算正确的索引
                episode_idx = (episode_num - batch_size + 1) + episode
                # 在第一个时间步 reward 前增加两列：当前 episode 的真实总 reward 和预测总 reward
                row = ", ".join(
                    [
                        str(episode_idx),
                        str(episode_total_reward),
                        str(episode_total_pred_reward),
                    ]
                    + step_cells
                )
                with open(self.individual_rewards_log_path, "a", encoding="utf-8") as f:
                    f.write(row + "\n")

            self.reward_save_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_reward_mixer.load_state_dict(self.reward_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        """
        新增
        """
        self.reward_mixer.cuda()
        self.target_reward_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        """
         * @author hyr
         * @modified 2025-11-25-17:17
         * @description 保存 Q 网络的同时保存 reward 网络
        """
        if self.args.reward_mixer:
            th.save(self.reward_mixer.state_dict(), "{}/reward_mixer.th".format(path))
            th.save(self.reward_optimizer.state_dict(), "{}/reward_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        """
         * @author hyr
         * @modified 2025-11-25-17:16
         * @description 加载 reward 网络
        """
        if self.args.reward_mixer:
            self.reward_mixer.load_state_dict(
                th.load(
                    "{}/reward_mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
            self.reward_optimizer.load_state_dict(
                th.load(
                    "{}/reward_opt.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

    """
     * @author hyr
     * @modified 2025-11-25-17:15
     * @description 单独保存 reward 网络
    """

    def save_reward_models(self, path):
        th.save(self.reward_mixer.state_dict(), "{}/reward_mixer.th".format(path))
        th.save(self.reward_optimizer.state_dict(), "{}/reward_opt.th".format(path))

    """
     * @author hyr
     * @modified 2025-11-25-19:16
     * @description 单独加载 reward 网络
    """

    def load_reward_models(self, path):
        self.reward_mixer.load_state_dict(
            th.load(
                "{}/reward_mixer.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.reward_optimizer.load_state_dict(
            th.load(
                "{}/reward_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
