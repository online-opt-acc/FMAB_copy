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
                # Hyperband,
                SuccessiveHalving,
                Uniform,
        )

from experiment_code.neural.train.hyperband import HyperbandRunner, ModelSampler

from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR100Handler
######################################################################
import multiobjective_opt.neural_net.models.pytorch_cifar_models as models_pull

def get_models():
    def get_n_params(model):
        return sum(p.numel() for p in model.parameters())
    
    cifar100_models = {}
    for model_class in dir(models_pull):
        if not "100" in model_class:
            continue
        model = getattr(models_pull, model_class)()
        if get_n_params(model) > 5e6:
            continue
        cifar100_models[model_class] = model
    return cifar100_models



######################################################################


def get_runner(
                mean_estimator_params,
                dataloader_cycled=True,
               dataloader_iters=50,
               datasets_path = None,
               train_hyperparams = None,
               eval_criterion = EvalCriterion.LOSS,
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

    data_loader = CIFAR100Handler(dataloader_cycled, dataloader_iters, root = datasets_path)
    train_hyperparams = TrainHyperparameters(**train_hyperparams)

    # set up environment
    env = NeuralArmEnv(
        models=models_list,
        model_params=model_params,
        model_names=model_names,
        data_loader=data_loader,
        eval_criterion=eval_criterion,
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
        case "Hyperband":
            model_sampler = ModelSampler(data_loader, eval_criterion, train_hyperparams, get_models_func=get_models)

            agent_runner = HyperbandRunner(model_sampler, 
                                           max_iter= None,
                                           eta = kwargs['hyperband_eta'],
                                           max_budget=kwargs['num_pulls'])
            return agent_runner, None, None

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

        if env is not None:
            alg_names = [arm.name for arm in env.arms]
            mlflow.log_param("alg_names", alg_names)

        result: NeuralRUNResult = runner.run(cfg.experiment.mab_params.num_pulls)

        serialized = cloudpickle.dumps(result)
        mlflow.log_text(serialized.decode("latin1"), "results.pkl")

if __name__ == "__main__":
    main()
    