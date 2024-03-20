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
from cartpole_q import plot_running_avg
from feature_transformer import FeatureTransformer
from mountain_car_rbf import plot_cost_to_go

from tqdm import tqdm



class HiddenLayer(nn.Module):
    def __init__(self, M1, M2, activation=nn.Tanh, use_bias=True, zeros=False):
        super(HiddenLayer, self).__init__()
        self.use_bias = use_bias
        self.W = nn.Linear(M1, M2, bias=use_bias)
        if zeros:
            nn.init.constant_(self.W.weight, 0)  # Initialize weights to zero
        else:
            nn.init.normal_(self.W.weight, mean=0, std=np.sqrt(2. / M1))  # Xavier initialization
        self.activation = activation()

    def forward(self, X):
        return self.activation(self.W(X))


class PolicyModel(nn.Module):
    def __init__(self, D, ft, hidden_layer_sizes=[512]):
        super(PolicyModel, self).__init__()
        self.ft = ft
        self.hidden_layers = nn.ModuleList()
        M1 = D
        for M2 in hidden_layer_sizes:
            self.hidden_layers.append(HiddenLayer(M1, M2))
            M1 = M2

        # Final layer for mean
        self.mean_layer = HiddenLayer(M1, 1, nn.Identity, use_bias=False, zeros=True)

        # Final layer for std deviation
        self.stdv_layer = HiddenLayer(M1, 1, nn.Softplus, use_bias=False, zeros=False)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, X):
        X = self.ft.transform(X)  # Assuming ft.transform is a numpy operation, might need adjustment for PyTorch
        Z = torch.from_numpy(X).float()
        for layer in self.hidden_layers:
            Z = layer(Z)
        mean = self.mean_layer(Z)
        stdv = self.stdv_layer(Z) + 1e-5
        return mean, stdv

    def sample_action(self, X):
        if not isinstance(X, np.ndarray):
            X = X[0]
        X = np.atleast_2d(X)
        X = torch.from_numpy(X).float()
        mean, stdv = self(X)
        normal_dist = torch.distributions.Normal(mean, stdv)
        action = torch.clamp(normal_dist.sample(), -1, 1)
        return action.item()

    def partial_fit(self, X, actions, advantages):
        if not isinstance(X, np.ndarray):
            X = X[0]
        X = torch.from_numpy(np.atleast_2d(X)).float()
        actions = torch.from_numpy(np.atleast_1d(actions)).float()
        advantages = torch.from_numpy(np.atleast_1d(advantages)).float()

        # We need the gradient of log (probability of action) given the current state and parameters * advantage
        self.optimizer.zero_grad()
        mean, stdv = self(X)
        normal_dist = torch.distributions.Normal(mean.view(-1), stdv.view(-1))
        # here we are sampling the probability of taking action a in the distribution parameterized by mean and stdv which
        # are the outputs of the neural network
        log_probs = normal_dist.log_prob(actions)
        loss = -(log_probs * advantages).mean() - 0.1 * normal_dist.entropy().mean()
        loss.backward()
        self.optimizer.step()


class ValueModel(nn.Module):
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        super(ValueModel, self).__init__()
        self.ft = ft
        self.layers = nn.ModuleList()
        M1 = D
        for M2 in hidden_layer_sizes:
            self.layers.append(HiddenLayer(M1, M2))
            M1 = M2

        self.final_layer = HiddenLayer(M1, 1, nn.Identity, use_bias=False, zeros=False)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-1)

    def forward(self, X):
        X = self.ft.transform(X)  # Adjust for PyTorch if necessary
        Z = torch.from_numpy(X).float()
        for layer in self.layers:
            Z = layer(Z)
        Y_hat = self.final_layer(Z)
        return Y_hat.view(-1)

    def partial_fit(self, X, Y):
        if not isinstance(X, np.ndarray):
            X = X[0]
        X = torch.from_numpy(np.atleast_2d(X)).float()
        Y = torch.from_numpy(np.atleast_1d(Y)).float()

        self.optimizer.zero_grad()
        Y_hat = self(X)
        loss = nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X[0]
        X = torch.from_numpy(np.atleast_2d(X)).float()
        with torch.no_grad():
            return self(X).numpy()


def play_one_td(env, pmodel, vmodel, gamma):
    # get the initial state
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    iterator = tqdm(range(2000), leave=True, position=0, desc="Playing Game")
    for _ in iterator:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info, _ = env.step([action])

        totalreward += reward

        # update the models
        V_next = vmodel.predict(observation)
        G = reward + gamma * V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

        iters += 1

    return totalreward, iters


def play_game(env, pmodel, vmodel, ft, num_episodes=1):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        totalreward = 0
        iters = 0

        while not done:
            env.render()  # Render the environment to visualize the agent

            if not isinstance(observation, np.ndarray):
                observation = observation[0]
            # Process the observation through feature transformer and policy model to get action
            X = np.atleast_2d(observation)
            action = pmodel.sample_action(X)

            # Take the action in the environment
            observation, reward, done, info, _ = env.step([action])
            totalreward += reward
            iters += 1

            if done:
                print(f"Episode {episode + 1}: Total Reward = {totalreward}, Steps = {iters}")
                break

    env.close()

def main(pmodel_path=None, vmodel_path=None, env=None, ft=None, gamma=0.95, num_episodes=50):
    if env is None:
        env = gym.make('MountainCarContinuous-v0')
    if ft is None:
        ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    if pmodel_path is None:
        pmodel = PolicyModel(D, ft, [512,128,64])
    else:
        pmodel = PolicyModel(D, ft, [512, 128, 64])
        pmodel.load_state_dict(torch.load(pmodel_path))

    if vmodel_path is None:
        vmodel = ValueModel(D, ft, [512, 128, 64])
    else:
        vmodel = ValueModel(D, ft, [512, 128, 64])
        vmodel.load_state_dict(torch.load(vmodel_path))
    try:

        if 'monitor' in sys.argv:
            filename = os.path.basename(__file__).split('.')[0]
            monitor_dir = './' + filename + '_' + str(datetime.now())
            env = wrappers.Monitor(env, monitor_dir)

        N = num_episodes
        totalrewards = np.empty(N)
        costs = np.empty(N)
        for n in range(N):
            totalreward, num_steps = play_one_td(env, pmodel, vmodel, gamma)
            totalrewards[n] = totalreward
            if n % 1 == 0:
                print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps,
                      "avg reward (last 100): %.1f" % totalrewards[max(0, n - 100):(n + 1)].mean())

            if n % 50000 == 0 :
                env = gym.make('MountainCarContinuous-v0', render_mode='human')
                play_game(env, pmodel, vmodel, ft, num_episodes=1)

        print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(totalrewards)
        plot_cost_to_go(env, vmodel)

        torch.save(pmodel.state_dict(), "pmodel.pth")
        torch.save(vmodel.state_dict(), "vmodel.pth")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        torch.save(pmodel.state_dict(), "pmodel.pth")
        torch.save(vmodel.state_dict(), "vmodel.pth")


if __name__ == '__main__':
    main(num_episodes=10000, pmodel_path="pmodel.pth", vmodel_path="vmodel.pth")