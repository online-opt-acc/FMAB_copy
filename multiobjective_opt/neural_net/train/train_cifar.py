import pickle
import os
import time
from pathlib import Path
from itertools import cycle

from matplotlib.dates import EPOCH_OFFSET
import pandas as pd
import hydra
from omegaconf import DictConfig
import mlflow

from tabulate import tabulate

from joblib import Parallel, delayed
from multiobjective_opt.neural_net.models.cifar_models import (
    CNN,
    MLP,
    CNNBatchNorm,
    CNNDropout,
    DeepCNN,
    ResNet18,
    SimpleLinearModel,
)
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from multiobjective_opt.neural_net.utils.funcs import train_model


def get_models():
    models = {
        "SimpleLinearModel": SimpleLinearModel(),
        "FullyConnectedModel": MLP(),
        "Conv2LayerModel": CNN(),
        "Conv3LayerModel": DeepCNN(),
        "ConvDropout": CNNDropout(),
        "ConvBatchNorm": CNNBatchNorm(),
        "ResNet": ResNet18(),
    }
    
    return models

def run_parallel_handler(parent_run_id,model_name, *args, **kwargs):
    with mlflow.start_run(run_name=model_name, nested=True, parent_run_id=parent_run_id) as child_run:
        # mlflow.set_tag("parent_run_id", parent_run_id)
        res = train_model(*args, **kwargs)
    return res
        
def run(parent_run_id, dataset_path, batch_size = 512, **kwargs):
    train_loader, test_loader = CIFAR10Handler(False, root = dataset_path).load_dataset(batch_size=batch_size)
    train_loader = train_loader.get_iterator()
    # Создание и обучение моделей
    models = get_models()
    devices = cycle(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    model_results = {}

    delayed_trainer = delayed(run_parallel_handler)
    parallelizer = Parallel(4)

    # model_names = list(models.keys())
    models = list(models.values())
    model_names = [m.__class__.__name__ for m in models]

    results = parallelizer(
        delayed_trainer(parent_run_id, model_name, model, train_loader, test_loader, verbose=False, device = device, **kwargs)
        for model, model_name, device in zip(models, model_names, devices)
    )

    for name, results in zip(model_names, results):
        model_results[name] = results

    return model_results

@hydra.main(version_base=None, config_path="./../../../conf/cv", config_name="config_full")
def main(cfg: DictConfig = None):
    exp_name = cfg.experiment.name

    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f'full_train:{time.strftime("%H:%M")}') as parent_run:
        mlflow.log_params(cfg)
        parent_run_id = parent_run.info.run_id
        res = run(parent_run_id = parent_run_id, dataset_path = cfg.paths.datasets_path, **cfg.experiment.full_train_params)
        print(res, "\n\n")
        data = []

        for model_name, model_res in res.items():
            accuracy = model_res["accuracy"]
            loss = model_res["loss"]
            running_loss = model_res["running_loss"]
            runtime = model_res["runtime"]
            ep = model_res["runned_epochs"]
            row = [model_name, running_loss, accuracy, loss, runtime, ep]
            data.append(row)

        headers = ["Model name", "Train loss", "Model Accuracy", "Model loss", "runtime", "epochs"]
        print(tabulate(data, headers=headers, floatfmt=".3f", tablefmt="heavy_outline"))

        df = pd.DataFrame(data, columns = headers)
        mlflow.log_table(df,str("run_results_table.json"),)
        mlflow.log_metric("runtime", runtime)


if __name__ == "__main__":
    main()
