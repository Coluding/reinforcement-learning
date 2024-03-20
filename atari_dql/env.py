import gym
import torch
import torchvision.transforms as tt
from PIL import Image
from typing import Tuple
import numpy as np

class AtariEnv:
    def __init__(self, env_name, render_mode="rgb_array", device="cpu"):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env_name = env_name
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape
        self.env = gym.make(env_name).unwrapped

        self.env.reset()
        self.current_screen = None
        self.done = False
        self.current_lives = None
        self.device = device
        self.state_transformer = StateTransformer(84, 84)

    def reset(self) -> torch.tensor:
        observation = self.env.reset()
        if not isinstance(observation, np.ndarray):
            observation = observation[0]

        return self.state_transformer.apply_transform(Image.fromarray(observation)).to(self.device)

    def step(self, action: int):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def take_action(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        observation, reward, done, info, lives = self.step(action)
        obs_tensor = self.state_transformer.apply_transform(Image.fromarray(observation)).to(self.device)
        return (obs_tensor, reward, done)



class StateTransformer:
    def __init__(self, img_height, img_width):
        self.transforms = tt.Compose([
            tt.Resize((img_height, img_width)),
            tt.Grayscale(),
            tt.ToTensor()
        ])

    def apply_transform(self, state: Image):
        return self.transforms(state).squeeze(0)