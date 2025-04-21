"""
here base class of arm is given
"""

from abc import abstractmethod
from typing import List, Union


from .arms import Reward, BaseArm


class BaseEnv:
    """
    Basic environment for different classes of problems
    """
    def __init__(self, n_actions, *args, **kwargs) -> None:
        self.n_actions = n_actions

    @property
    def num_actions(self):
        return self.n_actions

    @abstractmethod
    def pull(self, action: int) -> Union[Reward, None]:
        """
        pull arm and returns its reward
        """
        raise NotImplementedError

    def optimal_reward(self) -> int | float:
        """
        returns expected reward of the best arm
        """
        return None

    def action_reward(self, action: int) -> int | float:
        """
        returns expected reward of given action
        """
        return None


class ArmEnv(BaseEnv):
    def __init__(self, arms: List[BaseArm]):
        n_actions = len(arms)
        super().__init__(n_actions)
        self.arms = arms
    def pull(self, action) -> Union[Reward, None]:
        return self.arms[action].pull()
    def optimal_reward(self) -> Union[Reward, None]:
        return max(arm.ground_truth_reward() for arm in self.arms)
    
    def action_reward(self, action):
        return self.arms[action].ground_truth_reward()

    