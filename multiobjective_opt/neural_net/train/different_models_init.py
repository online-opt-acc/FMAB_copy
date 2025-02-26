import os
from pathlib import Path
import pickle

import hydra
from omegaconf import DictConfig

from joblib import Parallel, delayed
from multiobjective_opt.neural_net.utils.funcs import train_model

import multiobjective_opt.neural_net.pytorch_cifar_models as ptmodels
from multiobjective_opt.neural_net.utils import funcs
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from tabulate import tabulate

from multiobjective_opt.neural_net.train.different_models_mab import get_models

def run(data_path, epochs=100, batch_size = 512):
    train_loader, test_loader = CIFAR10Handler(False, root = data_path).load_dataset(batch_size=batch_size)
    train_loader = train_loader.get_iterator()
    # Создание и обучение моделей
    models = get_models()

    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    model_results = {}

    delayed_trainer = delayed(train_model)
    parallelizer = Parallel(4)

    # model_names = list(models.keys())
    model_names = [f"{m.__class__.__name__}_{i}" for i, m in enumerate(models)]

    results = parallelizer(
        delayed_trainer(model, train_loader, test_loader, epochs=epochs, verbose=False, device = device)
        for model, device in zip(models, devices)
    )

    for name, res in zip(model_names, results):
        model_results[name] = res

    return model_results

@hydra.main(version_base=None, config_path="./../../../conf/cv", config_name="config")
def main(cfg: DictConfig):
    savepath = cfg.paths.exp_savepath
    exp_name = cfg.full_train.name
    savepath = Path(savepath)
    if not savepath.exists():
        os.mkdir(savepath)

        
    res = run(**cfg.full_train.parameters)
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
    with open(savepath/f"{exp_name}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()