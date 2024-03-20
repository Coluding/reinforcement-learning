import torch
from strategy import *


class AtariAgent:
    def __init__(self, strategy: Strategy, num_actions: int, device: str = "cpu"):
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

        self.actions_taken = []
        self.states = []

    def select_action(self, state: torch.Tensor, q_model: torch.nn.Module, current_step: int) -> int:
        action = self.strategy.select_action(state, q_model, current_step)
        self.states.append(state)
        self.actions_taken.append(action)
        return action




