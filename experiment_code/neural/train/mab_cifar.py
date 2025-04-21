import time

import pandas as pd
import hydra
from tabulate import tabulate
from omegaconf import DictConfig
import mlflow

from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from multiobjective_opt.neural_net.utils.funcs import train_model

from multiobjective_opt.neural_net.train.train_cifar import get_models
import multiobjective_opt.neural_net.utils.funcs as funcs



def run(num_pulls=200, dataloader_cycled=True, dataloader_iters=50, datasets_path = None, train_hyperparams = None, **kwargs):
    dataloader_cycled = True
    models = get_models()
    model_names = models.keys()
    models_list = models.values()
    coeffs_list = [50 for i in range(len(models_list))]

    model_params = []
    for _ in models_list:
        model_params.append({})

    data_loader = CIFAR10Handler(dataloader_cycled, dataloader_iters, root = datasets_path)

    time_start = time.time()


    train_hyperparams = funcs.TrainHyperparameters(**train_hyperparams)
    neural_ucb = funcs.UCB_nets(models_list, \
                                model_params, \
                                model_names= model_names,\
                                data_loader=data_loader,\
                                coeffs=coeffs_list,\
                                train_hyperparams=train_hyperparams)
    ucb_ress = neural_ucb.ucb_train_models(sum_epochs=num_pulls)

    time_end = time.time() - time_start

    return ucb_ress, time_end


@hydra.main(version_base=None, config_path="./../../../conf/cv", config_name="config")
def main(cfg: DictConfig = None):
    exp_name = cfg.experiment.name

    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f'mab_train;{cfg.experiment.number};{time.strftime("%H:%M")}'):
        mlflow.log_params(cfg)

        ucb_res, runtime = run(datasets_path = cfg.paths.datasets_path, **cfg.experiment.mab_params)

        model_results, _, ucb_values = ucb_res
        mlflow.log_dict(ucb_values,str("ucb_values.json"))

        data = []

        for model in model_results:
            st = model.statistics
            accuracy = st.eval_results[-1]["accuracy"]
            loss = st.eval_results[-1]["loss"]
            confidence_interval = model.coeff / st.num_pulls**0.5
            row = [st.model_name, st.num_pulls, accuracy, loss, confidence_interval]
            data.append(row)
        headers = [
            "Model name",
            "model pulls",
            "Model Accuracy",
            "Model loss",
            "Confidence interval",
        ]
        print(f"{runtime=}")

        print(tabulate(data, headers=headers, floatfmt=".3f", tablefmt="heavy_outline"))
        
        df = pd.DataFrame(data, columns = headers)
        mlflow.log_table(df,str("run_results_table.json"),)
        mlflow.log_metric("runtime", runtime)
        data.append(runtime)
        
        # with open(savepath/f"{exp_name}.pkl", "wb") as f:
            # pickle.dump(data, f)

if __name__ == "__main__":
    main()
    