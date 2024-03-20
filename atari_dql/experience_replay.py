import numpy as np
import random
import torch


class PrioritizedReplayBuffer:
    def __init__(self, size=500000, frame_height=84, frame_width=84,
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
        self.actions = torch.empty(self.size, dtype=torch.float32)
        self.rewards = torch.empty(self.size, dtype=torch.float32)
        self.frames = torch.empty((self.size, self.frame_height, self.frame_width), dtype=torch.float32)
        self.terminal_flags = torch.empty(self.size, dtype=torch.bool)
        self.priorities = torch.ones(self.size, dtype=torch.float32, requires_grad=False)

        self.normalized_priorities = torch.zeros(self.size, dtype=torch.float32, requires_grad=False)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = torch.empty((self.batch_size, self.agent_history_length,
                                   self.frame_height, self.frame_width), dtype=torch.float32)
        self.new_states = torch.empty((self.batch_size, self.agent_history_length,
                                       self.frame_height, self.frame_width), dtype=torch.float32)
        self.indices = torch.empty(self.batch_size, dtype=torch.float32)

    def add_experience(self,
                       action: int,
                       frame: torch.Tensor,
                       reward: float,
                       terminal: bool):
        """
        Args:
            action: An integer-encoded action
            frame: One grayscale frame of the game of shape (self.frame_height, self.frame_width)
            reward: reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        # assign max priority to new experience so it is sampled at least once
        self.priorities[self.current] = self.priorities.max() if self.count > 0 else 1.0
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index: int) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("The replay memory is empty")
        if index < self.agent_history_length - 1:
            #raise ValueError(f'Index must be min {self.agent_history_length - 1}')
            index = 3
        if index > len(self.indices):
            index = len(self.indices)
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self) -> None:
        for i in range(self.batch_size):
            while True:
                try:
                    index = random.randint(self.agent_history_length, self.count - 1)
                    if index < self.agent_history_length:
                        continue
                    if index >= self.current and index - self.agent_history_length <= self.current:
                        continue
                    if self.terminal_flags[index - self.agent_history_length:index].any():
                        continue
                    break
                except Exception as e:
                    print(e)
            self.indices[i] = index

    def _sample_experience_based_on_priority(self, renormalize: bool = True) -> None:
        if renormalize:
            self.normalize_priorities()
        indices = torch.multinomial(self.normalized_priorities, self.batch_size, replacement=False)
        self.indices = indices

    def _get_valid_prioritized_indices(self) -> None:
        for i in range(self.batch_size):
            while True:
                index = random.randint(0, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def update_priorities(self, TD_errors: torch.Tensor) -> None:
        self.priorities[self.indices] = torch.abs(TD_errors) + 1e-5

    def normalize_priorities(self):
        self.normalized_priorities = torch.softmax(self.priorities, dim=0)

    def get_minibatch_prioritized(self):
        self._sample_experience_based_on_priority()
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx)
            self.new_states[i] = self._get_state(idx + 1)
        return (self.states, self.actions[self.indices], self.rewards[self.indices],
                self.new_states, self.terminal_flags[self.indices], self.indices)

    def save_buffer(self, file_path: str) -> None:
        """
        Saves the state of the replay buffer to a file.

        Args:
            file_path: The path (including file name) where the state should be saved.
        """
        state_dict = {
            'size': self.size,
            'frame_height': self.frame_height,
            'frame_width': self.frame_width,
            'agent_history_length': self.agent_history_length,
            'batch_size': self.batch_size,
            'count': self.count,
            'current': self.current,
            'actions': self.actions,
            'rewards': self.rewards,
            'frames': self.frames,
            'terminal_flags': self.terminal_flags,
            'priorities': self.priorities,
            'normalized_priorities': self.normalized_priorities,
            # Note: States and new_states are computed dynamically from frames,
            # so they don't necessarily need to be saved.
        }

        torch.save(state_dict, file_path)

    @classmethod
    def load_buffer(cls, file_path: str):
        """
        Loads the state of the replay buffer from a file and initializes a buffer instance with it.

        Args:
            file_path: The path (including file name) of the saved state to load.

        Returns:
            An initialized instance of PrioritizedReplayBuffer with the state loaded from the file.
        """
        state_dict = torch.load(file_path)

        # Initialize a new instance of PrioritizedReplayBuffer with the saved configuration
        buffer = cls(
            size=state_dict['size'],
            frame_height=state_dict['frame_height'],
            frame_width=state_dict['frame_width'],
            agent_history_length=state_dict['agent_history_length'],
            batch_size=state_dict['batch_size']
        )

        # Restore the saved state
        buffer.count = state_dict['count']
        buffer.current = state_dict['current']
        buffer.actions = state_dict['actions']
        buffer.rewards = state_dict['rewards']
        buffer.frames = state_dict['frames']
        buffer.terminal_flags = state_dict['terminal_flags']
        buffer.priorities = state_dict['priorities']
        buffer.normalized_priorities = state_dict['normalized_priorities']

        return buffer



