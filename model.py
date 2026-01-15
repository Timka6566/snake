import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.state_dict(), os.path.join(
            model_folder_path, file_name))

    def load(self, file_name='model.pth'):
        path = os.path.join('./model', file_name)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state, next_state, action, reward = [torch.unsqueeze(
                x, 0) for x in [state, next_state, action, reward]]
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * \
                    torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        self.criterion(target, pred).backward()
        self.optimizer.step()
