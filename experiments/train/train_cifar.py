import pickle

from experiments.models.cifar_models import (
    CNN,
    MLP,
    CNNBatchNorm,
    CNNDropout,
    DeepCNN,
    ResNet18,
    SimpleLinearModel,
)
from experiments.utils.dataset_prepare import CIFAR10Handler
from experiments.utils.funcs import train_model
from joblib import Parallel, delayed
from tabulate import tabulate


def main(verbose=False, epochs=100):
    train_loader, test_loader = CIFAR10Handler(False).load_dataset()
    train_loader = train_loader.get_iterator()
    # Создание и обучение моделей
    models = {
        "SimpleLinearModel": SimpleLinearModel(),
        "FullyConnectedModel": MLP(),
        "Conv2LayerModel": CNN(),
        "Conv3LayerModel": DeepCNN(),
        "ConvDropout": CNNDropout(),
        "ConvBatchNorm": CNNBatchNorm(),
        "ResNet": ResNet18(),
    }

    model_results = {}

    delayed_trainer = delayed(train_model)
    parallelizer = Parallel(7)

    # model_names = list(models.keys())
    models = list(models.values())
    model_names = [m.__class__.__name__ for m in models]

    results = parallelizer(
        delayed_trainer(model, train_loader, test_loader, epochs=epochs, verbose=verbose)
        for model in models
    )

    for name, res in zip(model_names, results):
        model_results[name] = res

    return model_results


if __name__ == "__main__":
    res = main(False)
    data = []

    for model_name, model_res in res.items():
        accuracy = model_res["accuracy"]
        loss = model_res["loss"]
        running_loss = model_res["running_loss"]
        row = [model_name, running_loss, accuracy, loss]
        data.append(row)

    headers = ["Model name", "Train loss", "Model Accuracy", "Model loss"]
    print(tabulate(data, headers=headers, floatfmt=".3f", tablefmt="heavy_outline"))
    with open("cifar_full_train.pkl", "wb") as f:
        pickle.dump(data, f)
