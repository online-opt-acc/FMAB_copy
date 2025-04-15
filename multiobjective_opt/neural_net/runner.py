from dataclasses import dataclass, asdict
from re import S
from time import time
from typing import List

from tqdm import tqdm
import mlflow

from multiobjective_opt.mab.agents import BaseAgent
from multiobjective_opt.mab.runner import RUNResult, RunAlgEnv
from multiobjective_opt.neural_net.mab_classes import (
    NeuralReward,
    NeuralArmEnv,
    EvalRez
)

from multiobjective_opt.utils.utils import flatten_dataclass

@dataclass
class NeuralRUNResult(RUNResult):
    test_history: List[EvalRez]

class NeuralRunner(RunAlgEnv):
    def __init__(self, environment: NeuralArmEnv, agent: BaseAgent):
        self.environment = environment
        self.agent = agent

    def run(self, max_steps) -> RUNResult:
        initial_test_values = []

        rewards_history = []
        actions_history = []
        test_history = []

        n_actions = self.environment.n_actions
        arms = self.environment.arms

        start_time = time()

        # evaluate arms without training
        for i in tqdm(range(n_actions), desc="Init eval"):
            test_rez: EvalRez = arms[i].test()
            test_rez.duration = time() - start_time

            initial_test_values.append(test_rez)

            # mlflow logs
        mlflow.log_dict({"init_test": [flatten_dataclass(a) for a in initial_test_values]}, "init_test.json")        
        del initial_test_values

        for i in tqdm(range(max_steps), desc="Training steps"):
            action = self.agent.get_action()

            reward: NeuralReward = self.environment.pull(action)
            reward.eval_rez.duration = time() - start_time

            self.agent.update(action, reward)

        # evaluation on test dataset
            test_rez: EvalRez = arms[action].test()
            test_rez.duration = time() - start_time

            mlflow.log_metrics(flatten_dataclass({"pull_rew": reward}),step=i)
            mlflow.log_metrics(flatten_dataclass({"test_rew": test_rez}),step=i)
            mlflow.log_metric("pulled_arm", action, step = i)

            rewards_history.append(reward)
            test_history.append(test_rez)
            actions_history.append(i)


        res = NeuralRUNResult(
                        n_actions=self.environment.n_actions,
                        rewards_history=rewards_history,
                        actions_history=actions_history,
                        test_history=test_history  
            )
        return res