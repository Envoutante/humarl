import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_actions
        self.unit_dim = args.unit_dim

        # 个体 reward 网络 - 所有智能体共享参数（不使用偏置项）
        # 移除最后一层 ReLU，允许输出负值，避免所有个体 reward 为 0
        self.individual_reward_net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.unit_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # 移除 ReLU，允许负值输出
        )

        # 全局 reward 聚合网络的超网络 - 基于全局 state 生成权重
        self.embed_dim = getattr(args, "mixing_embed_dim", 32)
        hypernet_layers = getattr(args, "hypernet_layers", 1)

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
            # 添加一个缩放因子网络，用于确保总奖励不超过个体奖励的和
            self.scale_net = nn.Linear(self.state_dim, 1)
        elif hypernet_layers == 2:
            hypernet_embed = getattr(args, "hypernet_embed", 64)
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
            # 添加一个缩放因子网络，用于确保总奖励不超过个体奖励的和
            self.scale_net = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, 1),
            )
        elif hypernet_layers > 2:
            raise ValueError("hypernet_layers > 2 is not implemented.")
        else:
            raise ValueError("hypernet_layers must be >= 1.")

    def forward(self, ep_batch):
        actions_onehot = ep_batch["actions_onehot"][:, :-1]
        state = ep_batch["state"][:, :-1]

        batch_size, seq_len, n_agents, _ = (
            actions_onehot.shape
        )  # actions_onehot: [batch_size, seq_len, n_agents, action_dim]

        # 计算个体reward（共享参数的网络）
        # QPLEX
        state_flat = state.reshape(
            -1, self.state_dim
        )  # [batch_size * seq_len, state_dim]
        unit_flat = state_flat[
            :, : self.unit_dim * self.n_agents
        ]  # [batch_size * seq_len, unit_dim * n_agents]
        unit_flat = unit_flat.reshape(
            -1, self.n_agents, self.unit_dim
        )  # [batch_size * seq_len, n_agents, unit_dim]
        unit_flat = unit_flat.reshape(
            -1, self.unit_dim
        )  # [batch_size * seq_len * n_agents, unit_dim]

        # 展平 state、action
        # 将全局 state 广播到每个 agent
        state_expanded = state.unsqueeze(2).expand(
            batch_size, seq_len, n_agents, self.state_dim
        )  # [batch_size, seq_len, n_agents, state_dim]
        state_flat = state_expanded.reshape(
            -1, self.state_dim
        )  # [batch_size * seq_len * n_agents, state_dim]
        actions_flat = actions_onehot.reshape(
            -1, self.action_dim
        )  # [batch_size * seq_len * n_agents, action_dim]

        # 拼接 state、action、unit_state
        agent_inputs = torch.cat(
            [state_flat, actions_flat, unit_flat], dim=-1
        )  # [batch_size * seq_len * n_agents, state_dim + action_dim + unit_dim]

        # 计算个体 reward（允许负值）
        individual_rewards_flat = self.individual_reward_net(
            agent_inputs
        )  # [batch_size * seq_len * n_agents, 1]
        individual_rewards = individual_rewards_flat.view(
            batch_size, seq_len, n_agents, 1
        )  # [batch_size, seq_len, n_agents, 1]

        # 使用超网络聚合全局 reward
        global_reward_pred = self._aggregate_global_reward(
            individual_rewards, state
        )  # [batch_size, seq_len, 1]

        return individual_rewards, global_reward_pred

    def _aggregate_global_reward(self, individual_rewards, state):
        """
        聚合个体 reward 为全局 reward
        使用归一化权重和缩放因子确保：global_reward <= sum(individual_rewards)
        这样可以避免子奖励和大于总奖励的情况
        """
        batch_size, seq_len, _, _ = individual_rewards.shape
        total_batches = batch_size * seq_len

        individual_rewards = individual_rewards.reshape(total_batches, 1, self.n_agents)
        states = state.reshape(total_batches, self.state_dim)

        # 计算个体 reward 的和（用于归一化）
        individual_sum = individual_rewards.sum(dim=2, keepdim=True)  # [total_batches, 1, 1]

        # 生成权重（使用 abs 确保非负）
        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)

        # 计算加权聚合（类似 QMIX）
        hidden = F.elu(torch.bmm(individual_rewards, w1))  # [total_batches, 1, embed_dim]

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # 计算未缩放的聚合值
        y_raw = torch.bmm(hidden, w_final)  # [total_batches, 1, 1]

        # 使用缩放因子确保总奖励不超过个体奖励的和
        # scale_factor 范围在 [0, 1] 之间，通过 sigmoid 实现
        scale_factor = torch.sigmoid(self.scale_net(states))  # [total_batches, 1]
        scale_factor = scale_factor.unsqueeze(1)  # [total_batches, 1, 1]

        # 最终输出：使用缩放因子限制在个体奖励和的范围内
        # 确保总奖励不超过个体奖励的和（绝对值）
        y = y_raw * scale_factor + individual_sum * (1 - scale_factor)
        
        # 进一步确保：总奖励的绝对值不超过个体奖励和的绝对值
        # 对于每个样本，如果 individual_sum > 0，则 y <= individual_sum
        # 如果 individual_sum < 0，则 y >= individual_sum
        # 使用逐元素的 clamp 确保单调性
        # 注意：torch.clamp 的 max/min 参数需要是标量或与输入形状兼容的张量
        y = torch.where(
            individual_sum > 0,
            torch.minimum(y, individual_sum),  # 如果和为正，限制上限
            torch.maximum(y, individual_sum)   # 如果和为负，限制下限
        )
        
        global_reward_pred = y.view(batch_size, seq_len, 1)

        return global_reward_pred
