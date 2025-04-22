import time
import numpy as np
import pandas as pd
import hydra
from tabulate import tabulate
from omegaconf import DictConfig
import mlflow

import cloudpickle
from multiobjective_opt.neural_net.mab_classes import (
    EvalCriterion,
    NeuralArmEnv,
    NeuralRewardEstimator,
    TrainHyperparameters
)
from multiobjective_opt.neural_net.runner import NeuralRunner, NeuralRUNResult
from multiobjective_opt.mab.agents import (
                UCB, 
                EpsGreedy,
                SuccessiveHalving,
                Uniform,
                Hyperband
        )


from experiment_code.neural.train_cifar import get_models
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler

def get_runner(
                mean_estimator_params,
                dataloader_cycled=True,
               dataloader_iters=50,
               datasets_path = None,
               train_hyperparams = None,
               mab_type = "UCB",
               **kwargs):
        
    # set up models
    models = get_models()
    model_names = models.keys()
    models_list = models.values()

    # ????
    model_params = []
    for _ in models_list:
        model_params.append({})

    data_loader = CIFAR10Handler(dataloader_cycled, dataloader_iters, root = datasets_path)
    train_hyperparams = TrainHyperparameters(**train_hyperparams)

    # set up environment
    env = NeuralArmEnv(
        models=models_list,
        model_params=model_params,
        model_names=model_names,
        data_loader=data_loader,
        eval_criterion=EvalCriterion.LOSS,
        train_hyperparams=train_hyperparams,
    )

    # set up agent
    n_actions = len(models)

        # set up mab agent class

    # initialize mab agent
    match mab_type:
        case "UCB":
            reward_estimator = NeuralRewardEstimator(
                n_actions=n_actions,
                **mean_estimator_params,
                )
            agent = UCB(
                    n_actions=n_actions,
                    reward_estimator=reward_estimator
                )
        case "EpsGreedy":
            assert "eps" in kwargs, "provide 'eps' parameter via 'experiment.mab_params.eps=1e-2'"
            reward_estimator = NeuralRewardEstimator(
                n_actions=n_actions,
                c = np.zeros(n_actions, float),
                **mean_estimator_params,
                )
            agent = EpsGreedy(
                    n_actions=n_actions,
                    reward_estimator=reward_estimator,
                    eps = kwargs["eps"]
                )
        case "Uniform":
            reward_estimator = NeuralRewardEstimator(
                n_actions=n_actions,
                c = np.zeros(n_actions, float),
                **mean_estimator_params,
                )
            
            agent = Uniform(
                    n_actions=n_actions,
                    reward_estimator=reward_estimator,
                    budget=kwargs['num_pulls']
                )
        case "SuccessiveHalving":
            reward_estimator = NeuralRewardEstimator(
                n_actions=n_actions,
                c = np.zeros(n_actions, float),
                **mean_estimator_params,
                )
            
            agent = SuccessiveHalving(
                    n_actions=n_actions,
                    reward_estimator=reward_estimator,
                    budget=kwargs['num_pulls']
                )
        case _:
            raise ValueError("there is no such type of agents")


    runner = NeuralRunner(
        environment=env,
        agent=agent
        )
    return runner, env, agent

@hydra.main(version_base=None, config_path="./../../conf/cv", config_name="config")
def main(cfg: DictConfig = None):
    exp_name = cfg.experiment.name

    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f'mab_train;{cfg.experiment.subexp_name};{time.strftime("%H:%M")}'):
        mlflow.log_params(cfg)

        runner, env, _ = get_runner(datasets_path = cfg.paths.datasets_path, **cfg.experiment.mab_params)

        alg_names = [arm.name for arm in env.arms]
        mlflow.log_param("alg_names", alg_names)

        result: NeuralRUNResult = runner.run(cfg.experiment.mab_params.num_pulls)

        serialized = cloudpickle.dumps(result)
        mlflow.log_text(serialized.decode("latin1"), "results.pkl")

if __name__ == "__main__":
    main()
    