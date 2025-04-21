import time
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
from multiobjective_opt.mab.agents import UCB, EpsGreedy


from multiobjective_opt.neural_net.train.train_cifar import get_models
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler

def get_runner(dataloader_cycled=True, 
               dataloader_iters=50, 
               datasets_path = None, 
               train_hyperparams = None, 
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
    coeff_scaler = 4.
    n_actions = len(models)
    reward_estimator = NeuralRewardEstimator(
        n_actions=n_actions, 
        coeff_scaler=coeff_scaler)
    
    agent = UCB(
        n_actions=n_actions, 
        reward_estimator=reward_estimator
    )
    runner = NeuralRunner(
        environment=env,
        agent=agent        
        )
    return runner
    


@hydra.main(version_base=None, config_path="./../../../conf/cv", config_name="config")
def main(cfg: DictConfig = None):
    exp_name = cfg.experiment.name

    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f'mab_train;{cfg.experiment.number};{time.strftime("%H:%M")}'):
        mlflow.log_params(cfg)

        runner = get_runner(datasets_path = cfg.paths.datasets_path, **cfg.experiment.mab_params)
        result: NeuralRUNResult = runner.run(cfg.experiment.mab_params.num_pulls)

        serialized = cloudpickle.dumps(result)
        mlflow.log_text(serialized.decode("latin1"), "results.pkl")

        # result.test_history


        # model_results, _, ucb_values = ucb_res
        # mlflow.log_dict(ucb_values,str("ucb_values.json"))

        # data = []

        # for model in model_results:
        #     st = model.statistics
        #     accuracy = st.eval_results[-1]["accuracy"]
        #     loss = st.eval_results[-1]["loss"]
        #     confidence_interval = model.coeff / st.num_pulls**0.5
        #     row = [st.model_name, st.num_pulls, accuracy, loss, confidence_interval]
        #     data.append(row)
        # headers = [
        #     "Model name",
        #     "model pulls",
        #     "Model Accuracy",
        #     "Model loss",
        #     "Confidence interval",
        # ]
        # print(f"{runtime=}")

        # print(tabulate(data, headers=headers, floatfmt=".3f", tablefmt="heavy_outline"))
        
        # df = pd.DataFrame(data, columns = headers)
        # mlflow.log_table(df,str("run_results_table.json"),)
        # mlflow.log_metric("runtime", runtime)
        # data.append(runtime)
        
        # with open(savepath/f"{exp_name}.pkl", "wb") as f:
            # pickle.dump(data, f)

if __name__ == "__main__":
    main()
    