from abc import abstractmethod

import numpy as np
from multiobjective_opt.mab.environment import Reward


class BaseRewardEstimator:
    """
    estimates sample mean as an reward estimation
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._mu = np.zeros((n_actions,), float)
        self._num_pulls = np.zeros((n_actions,), int)

    def get_estimations(self):
        """
        returns estimations of values, that to be maximized

        So if the aim is to minimize, then return - values
        """
        return self._mu
    @abstractmethod
    def update(self, action: int, reward: Reward):
        """
        update the reward estimation
        """
        rew_val = reward.value
        num_pulls= self._num_pulls[action]
        self._mu[action] = (num_pulls * self._mu[action] + rew_val)/(num_pulls + 1)
        self._num_pulls[action] += 1


class UCBClassicRewardEstimator(BaseRewardEstimator):
    def __init__(self, n_actions, T: float, c: float = 1.):
        super().__init__(n_actions)
        self.c = c
        self.T = T
        self.estimations = np.zeros((n_actions,), float)

    def get_estimations(self):
        return self.estimations

    def update(self, action: int, reward: Reward):
        super().update(action, reward)
        self.estimations[action] = self._mu[action] + \
            self.c * np.log(self.T)/ (self._num_pulls[action] ** 0.5)
        
