from abc import ABC, abstractmethod
import torch
import numpy as np


class Strategy(ABC):
    def __init__(self, start: int, end: int, decay: int, num_actions: int):
        self.start = start
        self.end = end
        self.decay = decay
        self.num_actions = num_actions

    def select_action(self, state: torch.Tensor, q_values: torch.nn.Module, current_step: int) -> int:
        rate = self.get_exploration_rate(current_step)
        if rate > torch.rand(1).item():
            return np.random.choice(range(self.num_actions))
        else:
            return torch.argmax(q_values(state)).item()

    @abstractmethod
    def get_exploration_rate(self, current_step: int) -> float:
        pass


class EpsilonGreedyStrategyExp(Strategy):
    def __init__(self, start: float, end: float, decay: float, num_actions: int):
        super(EpsilonGreedyStrategyExp, self).__init__(start, end, decay, num_actions)

    def get_exploration_rate(self, current_step) -> float:
        return (self.end + (self.start - self.end) * torch.exp(torch.tensor(-1. * current_step * self.decay))).item()


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, start, end, decay, num_actions: int, eps: float = 0.1):
        super(EpsilonGreedyStrategy, self).__init__(start, end, decay, num_actions)
        self.eps = eps

    def get_exploration_rate(self, current_step: int) -> float:
        return self.eps


class EpsilonGreedyLinear(Strategy):
    def __init__(self, start, end, decay, num_actions: int,
                 final_eps=None, startpoint=50000, kneepoint=1000000, final_knee_point=None) -> None:
        super(EpsilonGreedyLinear, self).__init__(start, end, decay, num_actions)
        self.final_eps = final_eps
        self.startpoint = startpoint
        self.kneepoint = kneepoint
        self.final_knee_point = final_knee_point

    def get_exploration_rate(self, current_step):
        if current_step < self.startpoint:
            return 1.
        mid_seg = self.end + \
                  np.maximum(0, (1 - self.end) - (1 - self.end) / self.kneepoint * (current_step - self.startpoint))
        if not self.final_eps:
            return mid_seg
        else:
            if self.final_eps and self.final_knee_point and (current_step < self.kneepoint):
                return mid_seg
            else:
                return self.final_eps + \
                    (self.end - self.final_eps) / (self.final_knee_point - self.kneepoint) * (
                            self.final_knee_point - current_step)
