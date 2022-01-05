from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


class RewardComputer(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = deepcopy(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.num_reward_signals = None

        self.cache = []  # (beams, reward)

    def set_number_reward_signals(self, num_reward_signals: int = 1):
        self.num_reward_signals = num_reward_signals

    def compute(self, state: np.array) -> Tuple[float, np.ndarray]:
        assert self.num_reward_signals is not None

        sorted_state = np.sort(state)
        for pair in self.cache:
            if np.all(pair[0] == sorted_state):
                return float(pair[1].mean()), pair[1]

        reward_signals = self.compute_reward(np.sort(state).tolist(), **self.config)
        # Ensure that the code is consistent
        assert reward_signals.shape == (self.num_reward_signals,)

        self.cache.append((np.sort(state), reward_signals))
        return float(reward_signals.mean()), reward_signals

    def load_cache(self, cache: List[Tuple[np.ndarray, float]]):
        self.cache = cache

    @property
    @classmethod
    @abstractmethod
    def DEFAULT_CONFIG(cls):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_reward() -> np.ndarray:
        pass
