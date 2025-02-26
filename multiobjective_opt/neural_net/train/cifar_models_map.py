import pickle
from dataclasses import asdict, dataclass, field
from typing import List
from time import time
from typing_extensions import runtime


from multiobjective_opt.neural_net.utils import funcs
from multiobjective_opt.neural_net.models.cifar_parametrized import ResNet
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
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



def main(num_pulls=500, dataloader_cycled=True, dataloader_iters=50):
    # loss_maximum = 40  # for cutting model coefficient
    dataloader_cycled = True

    models_list = get_models()
    coeffs_list = [70, 70, 70, 90,90,90, 110, 110, 110]
    model_params = []
    for _ in models_list:
        # models_list.append(models[name]())
        # coeffs_list.append(80)
        model_params.append({})

    data_loader = CIFAR10Handler(dataloader_cycled, dataloader_iters)

    time_start = time()
    neural_ucb = funcs.UCB_nets(models_list, \
                                model_params, \
                                data_loader, \
                                coeffs=coeffs_list,\
                                device="cuda:2")
    ucb_ress = neural_ucb.ucb_train_models(sum_epochs=num_pulls)

    time_end = time() - time_start

    return ucb_ress, time_end


import json

if __name__ == "__main__":
    ucb_res, runtime = main()
    model_results, losses, ucb_values = ucb_res

    print(ucb_res)
    with open("cifar_res_sched.json", "w") as f:
        json.dump(ucb_values, f)
        
    data = []

    for model in ucb_res[0]:
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
    
    data.append(runtime)

    with open("cifar_res_sched.pkl", "wb") as f:
        pickle.dump(data, f)