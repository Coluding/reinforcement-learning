import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import gym
from datetime import datetime
import random
from collections import deque
import torchvision.transforms as tt
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 1000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4  # Assuming a placeholder value, adjust based on your game's action space
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_FINAL = 0.5
EPSILON_DECAY = 500000


# Transform raw images for input into neural network
# 1) Convert to grayscale
# 2) Resize
# 3) Crop


class ImageTransformer:
    def __init__(self):
        self.transform = tt.Compose([
            tt.Grayscale(),
            tt.Resize((IM_SIZE, IM_SIZE)),
            tt.ToTensor()
        ])

    def apply_transform(self, image: np.ndarray):
        image = Image.fromarray(image)
        return self.transform(image)

    def apply_transform_to_numpy(self, image: np.ndarray):
        image = Image.fromarray(image)
        return np.squeeze( np.array(self.transform(image)), 0)



def update_state(state, obs_small):
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, 2), axis=2)


class ReplayMemory:
    def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number of transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer-encoded action
            frame: One grayscale frame of the game
            reward: reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if len(frame.shape) != 2:
            frame = np.squeeze(frame, 0)
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


class DQN(nn.Module):
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, device="cpu"):
        super(DQN, self).__init__()
        self.device = device
        conv_layers = []
        in_ch = K
        self.K = K
        out_ch = None
        for out_ch, kernel_size, stride in conv_layer_sizes:
            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
            conv_layers.append(conv_layer)
            conv_layers.append(nn.ReLU())
            in_ch = out_ch


        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        conv_out_size = self._get_conv_out((4, IM_SIZE, IM_SIZE))
        layers = []
        in_size = conv_out_size
        for M in hidden_layer_sizes:
            layer = nn.Linear(in_size, M)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_size = M

        self.fc = nn.Sequential(*layers)
        self.action_output = nn.Linear(in_size, K)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def _get_conv_out(self, shape):
        x = torch.randn(1, *shape)
        o = self.conv_layers(x)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.action_output(x)

    def copy_from(self, other):
        self.load_state_dict(other.state_dict())

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(K)
        else:
            return np.argmax(self.predict([x])[0])

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if X.shape[0] != self.K:
            X = X.permute(0, 3, 1, 2)
        return self.forward(X).cpu().detach().numpy()

    def compute_loss(self, states, actions, targets):
        """
        Update the model's weights
        :param states: np.array
        :param actions: np.array
        :param targets: np.array
        :return: loss
        """
        loss = nn.MSELoss()(self.forward(states).gather(1, actions.unsqueeze(-1)).squeeze(), targets.to(torch.float32))
        return loss

    def update(self, states, actions, targets):
        """
        Update the model's weights
        :param states: np.array
        :param actions: np.array
        :param targets: np.array
        :return: loss
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, targets)
        loss.backward()
        self.optimizer.step()
        return loss


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    # Sample experiences
    states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()

    # Calculate targets

    if not isinstance(next_states, torch.Tensor):
        next_states = torch.tensor(next_states, dtype=torch.float32)


    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=model.device)
    states_tensor = states_tensor.permute(0, 3, 1, 2)
    states_tensor = states_tensor.to(model.device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=model.device)
    next_states = next_states.permute(0, 3, 1, 2)
    next_states = next_states.to(model.device)
    next_Qs = target_model(next_states)
    next_Q = torch.max(next_Qs, dim=1).values
    targets = rewards_tensor + torch.logical_not(dones_tensor) * gamma * next_Q

    # Update model
    loss = model.update(states_tensor, actions_tensor, targets)
    return loss


def play_one(
        env,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        image_transformer,
        gamma,
        batch_size,
        epsilon,
        epsilon_change,
        epsilon_min):
    t0 = datetime.now()

    # Reset the environment
    obs = env.reset()
    if not isinstance(obs, np.ndarray):
        obs = obs[0]
    obs_small = image_transformer.apply_transform_to_numpy(obs)

    state = np.stack([obs_small] * 4, axis=2)
    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done = False
    while not done:

        # Update target network
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
            total_t, TARGET_UPDATE_PERIOD))

        # Take action
        action = model.sample_action(state, epsilon)
        obs, reward, done, _, info = env.step(action)
        if not isinstance(obs, np.ndarray):
            obs = obs[0]
        obs_small = image_transformer.apply_transform_to_numpy(obs)
        next_state = update_state(state, obs_small)

        # Compute total reward
        episode_reward += reward

        # Save the latest experience
        experience_replay_buffer.add_experience(action, obs_small, reward, done)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        # More debugging info
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

    return total_t, episode_reward, (
                datetime.now() - t0), num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def play_game_human(model, model_path=None, num_episodes=1, eps=0.1):
    env = gym.make('Breakout-v0', render_mode='human')
    model.device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path))
    image_transformer = ImageTransformer()

    for episode in range(num_episodes):
        observation = env.reset()[0]
        done = False
        totalreward = 0
        iters = 0
        if not isinstance(observation, np.ndarray):
            observation = observation[0]
        obs_small = image_transformer.apply_transform_to_numpy(observation)
        state = np.stack([obs_small] * 4, axis=2)
        while not done:
            env.render()
            action = model.sample_action(state, eps)
            print(action)
            obs, reward, done, _, info = env.step(action)
            if not isinstance(obs, np.ndarray):
                obs = obs[0]
            obs_small = image_transformer.apply_transform_to_numpy(obs)
            next_state = update_state(state, obs_small)
            state = next_state
            if done:
                print(f"Episode {episode + 1}: Total Reward = {totalreward}, Steps = {iters}")
                break

if __name__ == '__main__':
    # hyperparams and initialize stuff
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (128, 3, 1), (256, 2, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 3500
    total_t = 0
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)

    # epsilon
    # decays linearly until 0.1
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    # Create environment
    env = gym.envs.make("Breakout-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create models
    model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes, device=device)
    #play_game_human(model, 'atari_dqn_model_1.pt', num_episodes=5)
    target_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        device=device
    )
    model = model.to(device)
    target_model = target_model.to(device)

    image_transformer = ImageTransformer()



    print("Populating experience replay buffer...")
    obs = env.reset()

    iterator = tqdm(range(MIN_EXPERIENCES), desc="Populating experience replay buffer...")
    for i in iterator:
        action = np.random.choice(K)
        obs, reward, done, _, info = env.step(action)
        obs_small_tensor = image_transformer.apply_transform(obs[0])
        obs_small = obs_small_tensor.numpy()
        experience_replay_buffer.add_experience(action, obs_small, reward, done)

        if done:
            obs = env.reset()

    # Play a number of episodes and learn!
    t0 = datetime.now()
    iterator_episodes = tqdm(range(num_episodes), desc="Playing episodes...")
    for i in iterator_episodes:
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
            env,
            total_t,
            experience_replay_buffer,
            model,
            target_model,
            image_transformer,
            gamma,
            batch_sz,
            epsilon,
            epsilon_change,
            epsilon_min,
        )
        episode_rewards[i] = episode_reward

        last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
        print("Episode:", i,
              "Duration:", duration,
              "Num steps:", num_steps_in_episode,
              "Reward:", episode_reward,
              "Training time per step:", "%.3f" % time_per_step,
              "Avg Reward (Last 100):", "%.3f" % last_100_avg,
              "Epsilon:", "%.3f" % epsilon
              )
        sys.stdout.flush()
        print("Total duration:", datetime.now() - t0)

        torch.save(model.state_dict(), 'atari_dqn_model_1.pt')

    # Plot the smoothed returns
    #y = smooth(episode_rewards)
    #plt.plot(episode_rewards, label='orig')
    #plt.plot(y, label='smoothed')
    #plt.legend()
    #plt.show()