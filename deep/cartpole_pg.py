import gym
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
# Assuming cartpole_q.plot_running_avg exists and is compatible with PyTorch or is replaced with a compatible version
from cartpole_q import plot_running_avg


class HiddenLayer(nn.Module):
    def __init__(self, M1, M2, f=nn.Tanh(), use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.linear = nn.Linear(M1, M2, bias=use_bias)
        self.f = f

    def forward(self, X):
        return self.f(self.linear(X))


class PolicyModel(nn.Module):
    def __init__(self, D, K, hidden_layer_sizes):
        super().__init__()
        layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            layers.append(layer)
            M1 = M2
        # Final layer without an activation function
        layers.append(HiddenLayer(M1, K, f=nn.Softmax(dim=1), use_bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer(Z)
        return Z

    def sample_action(self, X):
        X = torch.from_numpy(np.atleast_2d(X).astype(np.float32))
        with torch.no_grad():
            p = self.forward(X)[0].numpy()
        return np.random.choice(len(p), p=p)


class ValueModel(nn.Module):
    def __init__(self, D, hidden_layer_sizes):
        super().__init__()
        layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            layers.append(layer)
            M1 = M2
        # Final layer with no activation function
        layers.append(HiddenLayer(M1, 1, f=lambda x: x))
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer(Z)
        return Z.view(-1)

def train_one_step(model, optimizer, inputs, targets):
    inputs = torch.from_numpy(np.atleast_2d(inputs).astype(np.float32))
    targets = torch.from_numpy(np.atleast_1d(targets).astype(np.float32))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def play_one_td(env, pmodel, vmodel, p_optimizer, v_optimizer, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        if not isinstance(observation, np.ndarray):
            observation = observation[0]
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info, _ = env.step(action)

        V_next = vmodel(torch.from_numpy(np.atleast_2d(observation).astype(np.float32)))[0]
        V_next = V_next.item() if not done else 0
        G = reward + gamma * V_next
        # baseline is used here!
        # the current value of the state is the baseline
        advantage = G - vmodel(torch.from_numpy(np.atleast_2d(prev_observation).astype(np.float32)))[0].item()

        # Update policy model
        p_optimizer.zero_grad()
        p_logits = pmodel(torch.from_numpy(np.atleast_2d(prev_observation).astype(np.float32)))
        # this is the back part of the policy gradient update equation
        # it defines the loss as the gradient of policy taken multiplied with the delta (advtange)
        p_loss = -torch.log(p_logits[0, action]) * advantage
        p_loss.backward()
        p_optimizer.step()

        # Update value model
        train_one_step(vmodel, v_optimizer, prev_observation, G)

        totalreward += reward
        iters += 1

    return totalreward

# The play_one_mc function would need similar adjustments as play_one_td, focusing on MC-specific logic.

def main():
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    p_optimizer = optim.Adam(pmodel.parameters(), lr=1e-2)
    v_optimizer = optim.Adam(vmodel.parameters(), lr=1e-2)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 1000
    totalrewards = np.empty(N)
    for n in range(N):
        totalreward = play_one_td(env, pmodel, vmodel, p_optimizer, v_optimizer, 0.99)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

if __name__ == '__main__':
    main()
