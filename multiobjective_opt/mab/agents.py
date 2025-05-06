from abc import ABCMeta, abstractmethod
from typing import Type

import numpy as np

from multiobjective_opt.mab.environment import Reward
from multiobjective_opt.mab.reward_estimators import BaseRewardEstimator
    

class BaseAgent(metaclass=ABCMeta):
    def __init__(self, n_actions, reward_estimator: BaseRewardEstimator):
        self._total_pulls: int = 0
        self.n_actions: int = n_actions
        self.reward_estimator = reward_estimator

        if not hasattr(self, "_name"):
            self._name = self.__class__.__name__
    
    @abstractmethod
    def get_action(self):
        """
        select action and return it
        """
        raise NotImplementedError

    def update(self, action: int, reward: Reward):
        """
        update accumulated parameters
        """
        self._total_pulls += 1
        self.reward_estimator.update(action, reward)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, arg):
        self._name = arg

class UCB(BaseAgent):
    def __init__(self, n_actions: int, reward_estimator: Type[BaseRewardEstimator]):
        super(UCB, self).__init__(n_actions, reward_estimator)
        self.init_pulls_order = np.random.permutation(n_actions)

    def get_action(self):
        if self._total_pulls < self.n_actions:
            return self.init_pulls_order[self._total_pulls]

        estimation = self.reward_estimator.get_estimations()
        return np.argmax(estimation)


class EpsGreedy(BaseAgent):
    def __init__(self, n_actions, reward_estimator, eps: float = 1e-2):
        super().__init__(n_actions, reward_estimator)
        self.eps = eps
        self.init_pulls_order = np.random.permutation(n_actions)

    def get_action(self):
        if self._total_pulls < self.n_actions:
            return self.init_pulls_order[self._total_pulls]

        if np.random.rand() < self.eps:
            # random action
            return np.random.choice(self.n_actions)
        # else use predicted action
        estimation = self.reward_estimator.get_estimations()
        return np.argmax(estimation)
    
class Uniform(BaseAgent):
    def __init__(self, n_actions: int, reward_estimator: BaseRewardEstimator, budget: int):
        super().__init__(n_actions, reward_estimator)
        self.pull_order = np.random.permutation(n_actions)
        self.budget = budget
    def get_action(self):
        if self._total_pulls < self.budget:
            pull_per_arm = self.budget//self.n_actions
            pos = self._total_pulls // pull_per_arm
            return self.pull_order[pos]
        # else greedily use the best action
        estimation = self.reward_estimator.get_estimations()
        return np.argmax(estimation)

class Hyperband(BaseAgent):
    def __init__(self, n_actions, reward_estimator, R, eta = 2):
        """
        :n_actions: number of actions
        :reward_estimator: estimator
        :R:  the budget,
        :eta: 
        """
        raise NotImplementedError()
        


class SuccessiveHalving(BaseAgent):
    def __init__(self, n_actions, reward_estimator, budget: int, eta= 2.):
        super().__init__(n_actions, reward_estimator)
        self.budget = budget
        # define values

        assert eta > 1

        self.eta = eta

        self._num_halving_iterations = int(np.ceil(np.log2(self.n_actions)/np.log2(self.eta)))
        self.active_arms = np.random.permutation(self.n_actions).tolist()

        self.iterator = self.actions_iterator()

    def _halving(self):
        estimation = self.reward_estimator.get_estimations()
        active_arm_estimations = estimation[self.active_arms]

        n_active_arms = len(self.active_arms)
        n_new_arms = int(np.floor(n_active_arms/self.eta))
        best_arm_ind = np.argsort(active_arm_estimations)[-n_new_arms:]

        self.active_arms = np.array(self.active_arms)[best_arm_ind].tolist()

    def actions_iterator(self):
        last_arm = None
        pulls_per_iteration = self.budget / self._num_halving_iterations

        for _ in range(self._num_halving_iterations):
            n_active_arms = len(self.active_arms)

            actions_per_arm = int(np.floor(pulls_per_iteration/n_active_arms))

            for arm in range(n_active_arms):
                for _ in range(actions_per_arm):
                    last_arm = self.active_arms[arm]
                    yield self.active_arms[arm]
            
            self._halving()

        while True:
            yield last_arm
        
    def get_action(self):
        return next(self.iterator)


# class Hyperband(BaseAgent):
#     def __init__(self, n_actions, reward_estimator, budget: int, eta = 2., stages = None):
#         self.logeta = lambda x: np.log( x ) / np.log( self.eta )

#         self.budget =  budget

#         if stages is None:
#             stages = int( self.logeta( self.max_iter ))