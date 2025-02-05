import pickle
from dataclasses import asdict, dataclass, field
from typing import List

from joblib import Parallel, delayed
from multiobjective_opt.neural_net.models.cifar_parametrized import ResNet
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from multiobjective_opt.neural_net.utils.funcs import train_model
from tabulate import tabulate


@dataclass
class ResNetParams:
    num_blocks: List = field(
        default_factory=lambda: [2, 2, 2, 2]
    )  # Количество блоков в каждом слое
    num_filters: int = 32  # Количество фильтров в начальном слое
    use_batchnorm: bool = False  # Использовать BatchNorm
    use_dropout: bool = False  # Использовать Dropout
    dropout_prob: float = 0.5  # Вероятность дропаут


def get_models():
    models = []

    a = ResNetParams(num_blocks=[8, 8, 8, 2], num_filters=64)
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(
        num_blocks=[2, 2, 2, 4],
        num_filters=64,
    )
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(
        num_blocks=[1, 1, 1, 1],
        num_filters=128,
    )
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    # with batch_norm
    a = ResNetParams(num_blocks=[8, 8, 8, 2], num_filters=64, use_batchnorm=True)
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(num_blocks=[2, 2, 2, 4], num_filters=64, use_batchnorm=True)
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(num_blocks=[1, 1, 1, 1], num_filters=128, use_batchnorm=True)
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    # + dropout
    a = ResNetParams(
        num_blocks=[8, 8, 8, 2], num_filters=64, use_batchnorm=True, use_dropout=True
    )
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(
        num_blocks=[2, 2, 2, 4], num_filters=64, use_batchnorm=True, use_dropout=True
    )
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    a = ResNetParams(
        num_blocks=[1, 1, 1, 1], num_filters=128, use_batchnorm=True, use_dropout=True
    )
    model = ResNet(**asdict(a))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    models.append(model)

    return models


def main(verbose=False, epochs=10):
    train_loader, test_loader = CIFAR10Handler(False).load_dataset(batch_size=256)
    train_loader = train_loader.get_iterator()
    # Создание и обучение моделей
    models = get_models()[:1]

    model_results = {}

    delayed_trainer = delayed(train_model)
    parallelizer = Parallel(7)

    # model_names = list(models.keys())
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
