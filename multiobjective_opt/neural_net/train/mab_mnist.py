from experiments.models.mnist_models import (
    MLP,
    Conv1LayerModel,
    Conv2LayerModel,
    GoodMNISTModel,
    SimpleLinearModel,
)
from experiments.utils import funcs
from experiments.utils.dataset_prepare import MNISTHandler
from tabulate import tabulate


def main(num_pulls=100, dataloader_cycled=True, dataloader_iters=100):
    # loss_maximum = 40  # for cutting model coefficient
    dataloader_cycled = True

    models = {
        "SimpleLinearModel": SimpleLinearModel,
        "FullyConnectedModel": MLP,
        "Conv1LayerModel": Conv1LayerModel,
        "Conv2LayerModel": Conv2LayerModel,
        "GoodMNISTModel": GoodMNISTModel,
    }

    models_list = []
    coeffs_list = []

    for name in models.keys():
        models_list.append(models[name]())
        coeffs_list.append(None)

    data_loader = MNISTHandler(dataloader_cycled, dataloader_iters)
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
