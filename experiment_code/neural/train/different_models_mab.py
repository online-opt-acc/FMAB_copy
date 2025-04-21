"""
run experiment with different models
"""

import pickle
import json
import os
from time import time
from torch import save
# from typing_extensions import runtime
from pathlib import Path
import pandas as pd

import hydra
from omegaconf import DictConfig
import mlflow

import multiobjective_opt.neural_net.pytorch_cifar_models as ptmodels
from multiobjective_opt.neural_net.utils import funcs
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from tabulate import tabulate

def get_models():
    model_names = [
        "cifar10_mobilenetv2_x0_5", 	 #70.88 815780   92.88
        "cifar10_shufflenetv2_x1_0",	 #72.39 1356104  93.79
        "cifar10_resnet56",           #72.63 861620      94.37
        "cifar10_mobilenetv2_x0_75",  #73.61 1483524     93.72
        "cifar10_shufflenetv2_x1_0",   #                  92.98
        "cifar10_resnet44"            #                  94.01
        ]
    
    models_list = []
    for name in model_names:
        model = getattr(ptmodels, name)()
        models_list.append(model)
    
    return models_list

def run(num_pulls=200, dataloader_cycled=True, dataloader_iters=50, datasets_path = None, train_hyperparams = None):
    dataloader_cycled = True
    models_list = get_models()

    coeffs_list = [50, 50, 50, 50, 50, 50]
    model_params = []
    for _ in models_list:
        # models_list.append(models[name]())
        # coeffs_list.append(80)
        model_params.append({})

    data_loader = CIFAR10Handler(dataloader_cycled, dataloader_iters, root = datasets_path)

    time_start = time()


    train_hyperparanms = funcs.TrainHyperparameters(**train_hyperparams)
    neural_ucb = funcs.UCB_nets(models_list, \
                                model_params, \
                                data_loader, \
                                coeffs=coeffs_list,\
                                train_hyperparams=train_hyperparanms)
    ucb_ress = neural_ucb.ucb_train_models(sum_epochs=num_pulls)

    time_end = time() - time_start

    return ucb_ress, time_end


@hydra.main(version_base=None, config_path="./../../../conf/cv", config_name="config")
def main(cfg: DictConfig):
    savepath = cfg.paths.exp_savepath
    exp_name = cfg.experiment.name


    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        mlflow.log_params(cfg)

        savepath = Path(savepath)
        if not savepath.exists():
            os.mkdir(savepath)

        ucb_res, runtime = run(**cfg.experiment.parameters)

        model_results, _, ucb_values = ucb_res
        with open(savepath/f"{exp_name}.json", "w") as f:
            json.dump(ucb_values, f)
        data = []

        for model in model_results:
            st = model.statistics
            accuracy = st.eval_results[-1]["accuracy"]
            loss = st.eval_results[-1]["loss"]
            confidence_interval = model.coeff / st.train_steps**0.5
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
    