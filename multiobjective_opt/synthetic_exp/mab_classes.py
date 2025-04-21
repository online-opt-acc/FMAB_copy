from dataclasses import dataclass
import numpy as np

from multiobjective_opt.mab.arms import Reward
from multiobjective_opt.mab.environment import BaseArm
from multiobjective_opt.mab.reward_estimators import BaseRewardEstimator
from multiobjective_opt.synthetic_exp.optimizers import BaseOptimizer


@dataclass
class FuncReward(Reward):
    confidence_bound: float
    

class FuncArm(BaseArm):
    def __init__(self, oracle: BaseOptimizer):
        self.optim = oracle

    def pull(self, *args, **kwargs):
        value = self.optim.step()
        conf_rad = self.optim.bounds()
        return FuncReward(value, conf_rad)
    
class FuncRewardEstimator(BaseRewardEstimator):
    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.estimations = np.zeros((n_actions,), float)
        self._function_values = np.ones((n_actions,), float) * np.inf
        
    def get_estimations(self):
        return -self.estimations
    
    def update(self, action, reward: FuncReward):
        self._function_values[action] = min(self._function_values[action], reward.value)
        self.estimations[action] = self._function_values[action] - reward.confidence_bound