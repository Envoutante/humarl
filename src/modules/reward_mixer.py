import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardMixer(nn.Module):
    def __init__(self, args, agent):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.reward_prediction_mode = getattr(args, "reward_prediction_mode", "residual")

        if self.reward_prediction_mode not in ("residual", "direct"):
            raise ValueError(
                "Unknown reward_prediction_mode: {}. Expected one of ['residual', 'direct'].".format(
                    self.reward_prediction_mode
                )
            )

        if not hasattr(agent, "fc1") or not hasattr(agent, "rnn"):
            raise ValueError("RewardMixer currently requires agent with fc1 and rnn")

        # 与 Q 网络保持同构（fc1 + rnn），但使用独立参数，不做共享。
        self.reward_fc1 = nn.Linear(agent.fc1.in_features, args.rnn_hidden_dim)
        self.reward_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # Reward 分支仅保留独立的最后一层，输出每个动作对应的 reward 值。
        self.reward_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, ep_batch):
        # 与 Q 网络输入完全一致：obs + last_action + agent_id。
        agent_inputs = self._build_mac_aligned_inputs(ep_batch)
        # actions: [bs, T, n_agents, 1]，后续用于从每个动作的 reward 中索引当前执行动作的 reward。
        actions = ep_batch["actions"][:, :-1].long()
        # r_tot: [bs, T, 1]，仅 residual 模式下用于构造均分基线 r_tot / N。
        global_rewards = ep_batch["reward"][:, :-1]

        batch_size, seq_len, n_agents, _ = agent_inputs.shape
        # 维度保护：确保 reward 分支输入与 reward fc1 期望维度一致。
        if agent_inputs.size(-1) != self.reward_fc1.in_features:
            raise ValueError(
                "RewardMixer input dim {} mismatches reward fc1 input dim {}".format(
                    agent_inputs.size(-1), self.reward_fc1.in_features
                )
            )
        # 按 agent 展平，逐时刻通过 reward 分支独立的 fc1+rnn 提取隐藏表示。
        agent_inputs = agent_inputs.reshape(batch_size * n_agents, seq_len, -1)

        hidden = self.reward_fc1.weight.new_zeros(
            batch_size * n_agents, self.args.rnn_hidden_dim
        )
        reward_logits_per_t = []
        for t in range(seq_len):
            x_t = F.relu(self.reward_fc1(agent_inputs[:, t]))
            hidden = self.reward_rnn(x_t, hidden)
            reward_logits_per_t.append(self.reward_head(hidden))

        # reward_logits: [bs, T, n_agents, n_actions]，表示每个动作的 reward 预测。
        reward_logits = torch.stack(reward_logits_per_t, dim=1)
        reward_logits = reward_logits.view(
            batch_size, n_agents, seq_len, self.n_actions
        )
        reward_logits = reward_logits.permute(0, 2, 1, 3).contiguous()

        selected_rewards = torch.gather(reward_logits, dim=3, index=actions)

        if self.reward_prediction_mode == "residual":
            # residual 模式：个体 reward = r_tot / N + residual。
            base_reward = (global_rewards / float(self.n_agents)).unsqueeze(2)
            individual_rewards = base_reward + selected_rewards
        else:
            # direct 模式：reward 网络直接预测个体 reward。
            individual_rewards = selected_rewards

        # 全局预测值 = sum_i(r_i) = r_tot + sum_i(residual_i)。
        global_reward_pred = self._aggregate_global_reward(individual_rewards)

        return individual_rewards, global_reward_pred

    def _build_mac_aligned_inputs(self, ep_batch):
        """按时间维构造与 BasicMAC._build_inputs 完全一致的输入特征。"""
        obs = ep_batch["obs"][:, :-1]
        bs, seq_len, n_agents, _ = obs.shape

        inputs = [obs]

        if self.args.obs_last_action:
            actions_onehot = ep_batch["actions_onehot"]
            prev_actions = torch.zeros_like(actions_onehot[:, :-1])
            prev_actions[:, 1:] = actions_onehot[:, :-2]
            inputs.append(prev_actions)

        if self.args.obs_agent_id:
            agent_ids = torch.eye(n_agents, device=obs.device).view(
                1, 1, n_agents, n_agents
            )
            agent_ids = agent_ids.expand(bs, seq_len, -1, -1)
            inputs.append(agent_ids)

        return torch.cat(inputs, dim=-1)

    def _aggregate_global_reward(self, individual_rewards):
        global_reward_pred = individual_rewards.sum(dim=2)

        return global_reward_pred
