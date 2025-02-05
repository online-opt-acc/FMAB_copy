import pickle

from multiobjective_opt.neural_net.models.cifar_models import MLP  # ResNet18,
from multiobjective_opt.neural_net.models.cifar_models import (
    CNN,
    CNNBatchNorm,
    CNNDropout,
    DeepCNN,
    SimpleLinearModel,
)
from multiobjective_opt.neural_net.utils import funcs
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from tabulate import tabulate


def main(models=None, num_pulls=100, dataloader_cycled=True, dataloader_iters=100):
    # loss_maximum = 40  # for cutting model coefficient
    dataloader_cycled = True

    if models is None:
        models = {
            "SimpleLinearModel": SimpleLinearModel,
            "FullyConnectedModel": MLP,
            "Conv2LayerModel": CNN,
            "Conv3LayerModel": DeepCNN,
            "ConvDropout": CNNDropout,
            "ConvBatchNorm": CNNBatchNorm,
            # "ResNet": ResNet18
        }

    models_list = []
    coeffs_list = []

    for name in models.keys():
        models_list.append(models[name]())
        coeffs_list.append(None)

    data_loader = CIFAR10Handler(dataloader_cycled, dataloader_iters)
    neural_ucb = funcs.UCB_nets(models_list, data_loader, coeffs_list)

    ucb_res = neural_ucb.ucb_train_models(sum_epochs=num_pulls)

    return ucb_res


if __name__ == "__main__":
    ucb_res = main()
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
    print(tabulate(data, headers=headers, floatfmt=".3f", tablefmt="heavy_outline"))

    with open("cifar_res.pkl", "wb") as f:
        pickle.dump(data, f)
