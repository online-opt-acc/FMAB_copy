

from abc import abstractmethod
from dataclasses import dataclass
from typing import Union


@dataclass
class Reward:
    value: float

class BaseArm:
    @abstractmethod
    def pull(self, *args, **kwargs) -> Union[Reward, None]:
        raise NotImplementedError()
    def ground_truth_reward(self) -> Union[Reward, None]:
        """
        returns ground truth reward of arm or 
        None if not specified
        """
        return None