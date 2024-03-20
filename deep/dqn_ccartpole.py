import gym
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from tqdm import tqdm
from rbf_nets import plot_running_avg


# global counter
global_iters = 0


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


class DQN(nn.Module):
    def __init__(self, D, K, hidden_layer_sizes, gamma, device='cpu'):
        super(DQN, self).__init__()
        layers = []
        for M in hidden_layer_sizes:
            layer = HiddenLayer(D, M)
            layers.append(layer)
            D = M

        layers.append(nn.Linear(D, K))
        self.layers = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = 10000
        self.min_experiences = 100
        self.batch_size = 32
        self.gamma = gamma
        self.K = K
        self.device = device


    def forward(self, X):
        return self.layers(X)

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float().to(self.device)
        return self.forward(X).cpu().detach().numpy()

    def copy_from(self, other):
        self.load_state_dict(other.state_dict())

    def train_model(self, target_network):
        if len(self.experience['s']) < self.min_experiences:
              # don't do anything if we don't have enough experience
              return

              # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)


        # Q-Leanring update r + gamma * max(next_Q))
        next_Q = torch.max(target_network(next_states_tensor), dim=1).values
        targets = torch.tensor([r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)], device=self.device)

        pred = self.forward(states_tensor)
        loss = nn.MSELoss()(pred.gather(1, actions_tensor.unsqueeze(-1)).squeeze(), targets.to(torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    def sample_action(self, x, eps):
        if not isinstance(x, np.ndarray):
            x = x[0]
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])




def play_one(env, model, tmodel, eps, gamma, copy_period):
  global global_iters
  observation = env.reset()[0]
  done = False
  totalreward = 0
  iters = 0
  iterator = tqdm(range(2000), leave=True, position=0, desc="Playing Game")
  for _ in iterator:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info, _ = env.step(action)

    totalreward += reward
    if done:
        reward = -200


    # update the model
    model.add_experience(prev_observation, action, reward, observation, done)
    model.train_model(tmodel)

    iters += 1
    global_iters += 1

    if global_iters % copy_period == 0:
        iterator.set_description("Copying model")
        tmodel.copy_from(model)

    iterator.set_description(f"Playing Game")

  return totalreward


def main(model_path = None):
  env = gym.make('CartPole-v0')
  gamma = 0.99
  copy_period = 200
  best_reward = 0

  D = len(env.observation_space.sample())
  K = env.action_space.n
  sizes = [200,200]
  model = DQN(D, K, sizes, gamma, device='cuda' if torch.cuda.is_available() else 'cpu')
  if model_path is not None:
    model.load_state_dict(torch.load(model_path))
  model = model.to(model.device)
  tmodel = DQN(D, K, sizes, gamma, device='cuda' if torch.cuda.is_available() else 'cpu')
  tmodel = tmodel.to(tmodel.device)

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 5000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
    totalrewards[n] = totalreward
    if totalreward > best_reward:
        best_reward = totalreward
        torch.save(model.state_dict(), 'best_q_model.pth')
        print("saved new best model with reward of " + str(best_reward) + " in step " + str(global_iters))

    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


def play_game_human(model, model_path, num_episodes=1):
    env = gym.make('CartPole-v0', render_mode='human')
    model.load_state_dict(torch.load(model_path))


    for episode in range(num_episodes):
        observation = env.reset()[0]
        done = False
        totalreward = 0
        iters = 0
        while not done:
            env.render()  # Render the environment to visualize the agent

            if not isinstance(observation, np.ndarray):
                observation = observation[0]
            # Process the observation through feature transformer and policy model to get action
            X = np.atleast_2d(observation)
            action = model.sample_action(X, 0)

            # Take the action in the environment
            observation, reward, done, info, _ = env.step(action)

            if done:
                print(f"Episode {episode + 1}: Total Reward = {totalreward}, Steps = {iters}")
                break


if __name__ == '__main__':
    model = DQN(4, 2, [200, 200], 0.99, device='cpu' if torch.cuda.is_available() else 'cpu')
    play_game_human(model, 'best_q_model.pth', num_episodes=1)
    #main('best_q_model.pth')