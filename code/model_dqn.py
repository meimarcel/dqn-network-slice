import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from memory import PrioritizedReplayMemory


class QNet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(n_observations, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, n_actions)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.fc4(out)

        return out


class DQN:
    def __init__(self, num_input, num_actions, device):
        self.device = device
        self.lr = 1e-4
        self.target_net_update_freq = 500
        self.experience_replay_size = 50000
        self.batch_size = 128
        self.gamma = 0.99
        self.update_count = 0

        self.num_feats = num_input
        self.num_actions = num_actions

        self.train_hist = {"loss": []}

        self.memory = PrioritizedReplayMemory(self.experience_replay_size)
        self.cur_memory = None

        self.Q_model = QNet(self.num_feats, self.num_actions).to(self.device)
        self.Q_T_model = QNet(self.num_feats, self.num_actions).to(self.device)
        self.Q_T_model.load_state_dict(self.Q_model.state_dict())

        self.optimizer = optim.Adam(self.Q_model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def append_to_replay(self, state, action, reward, next_state):
        self.memory.push((state, action, reward, next_state))

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.Q_model.state_dict(), f"{path}/Q_model_{epoch}.dump")
        pickle.dump(self.memory, open(f"{path}/exp_replay_agent_{epoch}.dump", "wb"))

    def load(self, path, epoch):
        fname_Q_model = f"{path}/Q_model_{epoch}.dump"
        fname_replay = f"{path}/exp_replay_agent_{epoch}.dump"

        if os.path.isfile(fname_Q_model):
            self.Q_model.load_state_dict(torch.load(fname_Q_model))
            self.Q_T_model.load_state_dict(torch.load(fname_Q_model))

        if os.path.isfile(fname_replay):
            self.memory = pickle.load(open(fname_replay, "rb"))

    def plot_loss(self):
        plt.figure(2)
        plt.clf()
        plt.title("Training loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.plot(self.train_hist["loss"], "r")
        plt.legend(["loss"])
        plt.pause(0.001)

    def get_minibatch(self):
        transitions, indexes, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float32)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.int64)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float32)
        batch_next_state_indexs = torch.tensor(
            tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool
        )
        batch_next_state = torch.tensor(
            np.array(tuple(filter(lambda x: x is not None, batch_next_state))), device=self.device, dtype=torch.float32
        )

        return batch_state, batch_action, batch_reward, batch_next_state_indexs, batch_next_state, indexes, weights

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            with torch.no_grad():
                for target_param, param in zip(self.Q_T_model.parameters(), self.Q_model.parameters()):
                    target_param.data.copy_(param.data)

    def update(self):
        if len(self.memory._storage) < self.batch_size:
            return None

        batch_state, batch_action, batch_reward, batch_next_state_indexes, batch_next_state, indexes, weights = (
            self.get_minibatch()
        )

        current_q_values_samples = self.Q_model(batch_state)
        current_q_values_samples = current_q_values_samples.gather(1, batch_action.unsqueeze(-1))
        current_q_values_samples = current_q_values_samples.squeeze(-1)

        max_next_action_q_value = torch.zeros(self.batch_size, device=self.device)
        if batch_next_state.shape[0] > 0:
            with torch.no_grad():
                max_next_action_q_value[batch_next_state_indexes] = self.Q_T_model(batch_next_state).max(1).values

        target_q_values = (max_next_action_q_value * self.gamma) + batch_reward

        td_error = torch.abs(current_q_values_samples - target_q_values).detach()
        loss = torch.mean((current_q_values_samples - target_q_values) ** 2 * weights)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.Q_model.parameters(), 100)

        self.optimizer.step()

        self.train_hist["loss"].append(loss.item())

        self.update_target_model()

        self.memory.update_priorities(indexes, td_error.cpu().numpy())

        # print('Current Q value', current_q_values_samples.mean(1))
        # print('Expected Q value', batch_reward.mean(1))
