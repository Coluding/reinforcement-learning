import torch
import torch.optim as optim
from strategy import *
from env import *
from experience_replay import *
from dqns import *
from tqdm import tqdm
import logging
from datetime import datetime
from agent import *
import gc

logging.basicConfig(level=logging.INFO)


class DQLAlgorithms:
    def __init__(self,
                 atari_env: AtariEnv,
                 experience_replay_buffer: PrioritizedReplayBuffer,
                 q_model: torch.nn.Module,
                 target_q_model: torch.nn.Module,
                 gamma: float,
                 learning_rate: float,
                 batch_size: int,
                 agent: AtariAgent,
                 target_update_period: int,
                 device: str = "cpu"):
        self.agent = agent
        self.num_actions = strategy.num_actions
        self.device = device
        self.atari_env = atari_env
        self.experience_replay_buffer = experience_replay_buffer
        self.q_model = q_model
        self.target_q_model = target_q_model
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.q_model.parameters(), lr=learning_rate, eps=1e-2)
        self.global_step = 0
        self.target_update_period = target_update_period

        self.tracker_dict = {
            "loss": [],
            "reward": [],
            "episode_loss": [],
            "episode_reward": []
        }
        self.actions_taken = []
        self.states = []

    def fill_experience_replay_buffer(self, num_episodes: int):
        """
        Fill experience replay buffer with random actions
        :param num_episodes: Number of episodes to play. Each episode is played until the game is over
        :return:
        """
        iterator = tqdm(range(num_episodes), desc="Filling Experience Replay Buffer")
        for _ in iterator:
            state = self.atari_env.reset()
            done = False
            while not done:
                action = np.random.choice(range(self.num_actions))
                next_state, reward, done = self.atari_env.take_action(action)
                self.experience_replay_buffer.add_experience(action, state, reward, done)
                state = next_state
                if done:
                    break

    def play_one(self):
        observation: torch.Tensor = self.atari_env.reset()
        state: torch.Tensor = torch.stack([observation, observation, observation, observation], dim=0)
        done = False
        loss = 0

        total_time_training = 0
        num_steps_in_episode = 0
        episode_reward = 0

        done = False
        while not done:
            if self.global_step % self.target_update_period == 0:
                self.target_q_model.copy_weights(self.q_model)
                logging.info(f"Target Q Model Updated in Global Step {self.global_step}")

            state = state.to(self.device)
            action = self.agent.select_action(state, self.q_model, self.global_step)
            observation, reward, done = self.atari_env.take_action(action)
            self.experience_replay_buffer.add_experience(action, observation, reward, done)
            observation = observation.to(self.device)
            next_state = self.update_state(state, observation.unsqueeze(0))
            episode_reward += reward
            self.tracker_dict["reward"].append(reward)
            t0_2 = datetime.now()
            dt = datetime.now() - t0_2
            step_loss = self.update_model()
            loss += step_loss
            total_time_training += dt.total_seconds()
            num_steps_in_episode += 1
            state = next_state

            self.global_step += 1


        self.tracker_dict["episode_loss"].append(loss)
        self.tracker_dict["episode_reward"].append(episode_reward)
        return loss, episode_reward, num_steps_in_episode,

    def update_state(self, state: torch.Tensor, new_frame: torch.Tensor):
        return torch.cat((state[1:, :, :], new_frame), dim=0)

    def transfer_to_device(self, *args):
        return [arg.to(self.device) for arg in args]

    def update_model(self):
        allocation_start = torch.cuda.memory_allocated()
        states, actions, rewards, next_states, dones, experience_indices = self.experience_replay_buffer.get_minibatch_prioritized()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        next_Qs = self.target_q_model(next_states)
        next_Q = torch.max(next_Qs, dim=1)[0]
        targets = rewards + self.gamma * next_Q * torch.logical_not(dones)
        predicted_Qs = self.q_model(states)

        TD_errors = targets - predicted_Qs[range(self.batch_size), actions.to(int)]
        self.experience_replay_buffer.update_priorities(TD_errors.to("cpu"))

        loss = torch.mean(TD_errors ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.tracker_dict["loss"].append(loss.item())
        del next_Qs
        del next_Q
        del predicted_Qs
        del TD_errors
        del next_states
        del actions
        del rewards
        gc.collect()
        torch.cuda.empty_cache()
        allocation_end = torch.cuda.memory_allocated()

        return loss.item()

    def train_agent(self, num_episodes: int):
        for episode in tqdm(range(num_episodes)):
            loss, reward, num_steps = self.play_one()
            logging.info(f"Episode {episode} completed with loss {loss}, reward {reward}, and num steps {num_steps}")
            print(f"Episode {episode} completed with loss {loss}, reward {reward}, and num steps {num_steps}")

        return self.tracker_dict

    def save_state(self, path: str):
        torch.save(self.q_model.state_dict())

    def load_state(self, path: str):


if __name__ == "__main__":
    env = AtariEnv("Breakout-v4")
    strategy = EpsilonGreedyStrategyExp(1, 0.01, 0.001, env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = AtariAgent(strategy, 4, device)
    #experience_replay_buffer = PrioritizedReplayBuffer(50000, 84, 84, 4)
    experience_replay_buffer = PrioritizedReplayBuffer.load_buffer("replay_10000.bf")
    q_model = DQNCNN().to(device)
    target_q_model = DQNCNN().to(device)
    dql = DQLAlgorithms(env, experience_replay_buffer,
                        q_model, target_q_model,
                        0.99, 0.00025, 16,
                        agent, 1000, device=device)
    #dql.fill_experience_replay_buffer(10000)
    experience_replay_buffer.batch_size = 16
    #experience_replay_buffer.save_buffer("./replay_10000.bf")
    result = dql.train_agent(20000)
    print(result)
    print("Done")