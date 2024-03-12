import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import torch
from torch.optim import SGD
import torch.nn as nn
GAMMA = 0.99
ALPHA = 0.0005


class Approximator(nn.Module):
    def __init__(self, dim: int = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.tensor):
        return self.model(x)


    def __add__(self, other):
        self.model

def epsilon_greedy(model, s, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        if isinstance(values[0], torch.Tensor):
            values = [x.detach().numpy() for x in values]
        return np.argmax(values)
    else:
        return model.env.action_space.sample()


def gather_samples(env, n_episodes=10000):
    samples = []
    for _ in range(n_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)

            s, r, done, truncated, info = env.step(a)
    return samples


class Model:
    def __init__(self, env, ann: bool = False):
        # fit the featurizer to data
        self.env = env
        samples = gather_samples(env)
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.ann = ann
        if ann:
            self.w = Approximator(100)
            self.optimizer = SGD(self.w.parameters(), lr=ALPHA)
        else:
            # initialize linear model weights
            self.w = np.zeros(dims)


    def predict(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]

        if self.ann:
            x = torch.tensor(x, dtype=torch.float32)
            return self.w(x)
        else:
            return x @ self.w

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in range(self.env.action_space.n)]

    def grad(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x

    def predict_and_grad_ann(self, s, a, target: torch.Tensor):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        pred = self.w(x)
        error = target - pred
        error.backward()
        return error.detach().numpy(), x.grad.detach().numpy()

    def train_ann(self, s, a, target: torch.Tensor):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        pred = self.w(x)
        self.optimizer.zero_grad()
        loss = (target - pred) ** 2
        loss.backward()
        self.optimizer.step()




def test_agent(model, env, n_episodes=20):
    reward_per_episode = np.zeros(n_episodes)
    for it in range(n_episodes):
        done = False
        truncated = False
        episode_reward = 0
        s, info = env.reset()
        while not (done or truncated):
            a = epsilon_greedy(model, s, eps=0)
            s, r, done, truncated, info = env.step(a)
            episode_reward += r
        reward_per_episode[it] = episode_reward
    return np.mean(reward_per_episode)


def watch_agent(model, env, eps):
    done = False
    truncated = False
    episode_reward = 0
    s, info = env.reset()
    while not (done or truncated):
        a = epsilon_greedy(model, s, eps=eps)
        s, r, done, truncated, info = env.step(a)
        episode_reward += r
    print("Episode reward:", episode_reward)


if __name__ == '__main__':
    # instantiate environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    ann = True
    model = Model(env, ann)
    reward_per_episode = []

    # watch untrained agent
    watch_agent(model, env, eps=0)

    # repeat until convergence
    n_episodes = 1500
    for it in range(n_episodes):
        s, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            a = epsilon_greedy(model, s)
            s2, r, done, truncated, info = env.step(a)

            # get the target
            if done:
                target = r
            else:

                values = model.predict_all_actions(s2)
                if isinstance(values[0], torch.Tensor):
                    values = [x.detach().numpy() for x in values]
                # apply Q learning here
                target = r + GAMMA * np.max(values)

            if ann:
                model.train_ann(s, a, torch.tensor(target))
            else:
                # update the model
                g = model.grad(s, a)
                err = target - model.predict(s, a)
                model.w += ALPHA * err * g

            # accumulate reward
            episode_reward += r

            # update state
            s = s2

        if (it + 1) % 50 == 0:
            print(f"Episode: {it + 1}, Reward: {episode_reward}")

        # early exit
        if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
            print("Early exit")
            break

        reward_per_episode.append(episode_reward)

    # test trained agent
    test_reward = test_agent(model, env)
    print(f"Average test reward: {test_reward}")

    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()

    # watch trained agent
    env = gym.make("CartPole-v1", render_mode="human")
    watch_agent(model, env, eps=0)