from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from multiobjective_opt.mab.arms import Reward
from multiobjective_opt.utils.utils import get_fig_set_style

from .agents import BaseAgent
from .environment import BaseEnv


@dataclass
class RUNResult:
    n_actions: int
    rewards_history: List[Reward]
    actions_history: List[int]

class RunAlgEnv:
    def __init__(self, environment: BaseEnv, agent: BaseAgent):
        self.environment = environment
        self.agent = agent

    def run(self, max_steps) -> RUNResult:
        rewards_history = []
        actions_history = []

        for _ in range(max_steps):
            action = self.agent.get_action()
            reward: Reward = self.environment.pull(action)
            self.agent.update(action, reward)
            
            rewards_history.append(reward)
            actions_history.append(action)
        res = RUNResult(
                        n_actions=self.environment.n_actions,
                        rewards_history=rewards_history,
                        actions_history=actions_history
                        )
        return res

        