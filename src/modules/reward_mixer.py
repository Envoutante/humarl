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

        # 个体 reward 网络 - 所有智能体共享参数
        self.individual_reward_net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.unit_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 全局 reward 聚合网络的超网络 - 基于全局 state 生成权重和偏置
        self.embed_dim = getattr(args, "mixing_embed_dim", 32)
        hypernet_layers = getattr(args, "hypernet_layers", 1)

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
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
        elif hypernet_layers > 2:
            raise ValueError("hypernet_layers > 2 is not implemented.")
        else:
            raise ValueError("hypernet_layers must be >= 1.")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obs, actions_onehot, state):
        batch_size, seq_len, n_agents, _ = obs.shape

        # 计算个体reward（共享参数的网络）
        # QPLEX
        state_flat = state.reshape(-1, self.state_dim)
        unit_flat = state_flat[:, : self.unit_dim * self.n_agents]
        unit_flat = unit_flat.reshape(-1, self.n_agents, self.unit_dim)
        unit_flat = unit_flat.reshape(-1, self.unit_dim)

        # 展平 state、action
        # 将全局 state 广播到每个 agent
        state_expanded = state.unsqueeze(2).expand(batch_size, seq_len, n_agents, self.state_dim)
        state_flat = state_expanded.reshape(-1, self.state_dim)
        actions_flat = actions_onehot.reshape(-1, self.action_dim)

        # 拼接 state、action、unit_state
        agent_inputs = torch.cat([state_flat, actions_flat, unit_flat], dim=-1)

        individual_rewards_flat = self.individual_reward_net(agent_inputs)
        individual_rewards = individual_rewards_flat.view(batch_size, seq_len, n_agents, 1)

        # 使用超网络聚合全局 reward
        global_reward_pred = self._aggregate_global_reward(individual_rewards, state)

        return individual_rewards, global_reward_pred

    def _aggregate_global_reward(self, individual_rewards, state):
        batch_size, seq_len, _, _ = individual_rewards.shape
        total_batches = batch_size * seq_len

        individual_rewards = individual_rewards.reshape(total_batches, 1, self.n_agents)
        states = state.reshape(total_batches, self.state_dim)

        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(individual_rewards, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + v
        global_reward_pred = y.view(batch_size, seq_len, 1)

        return global_reward_pred

