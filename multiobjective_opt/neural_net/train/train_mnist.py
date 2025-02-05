from multiobjective_opt.neural_net.models.mnist_models import (
    MLP,
    Conv1LayerModel,
    Conv2LayerModel,
    GoodMNISTModel,
    SimpleLinearModel,
)
from multiobjective_opt.neural_net.utils.dataset_prepare import load_mnist
from multiobjective_opt.neural_net.utils.funcs import train_model
from tqdm import tqdm


def main(verbose=False, epochs=10):
    train_loader, test_loader = load_mnist()

    # Создание и обучение моделей
    models = {
        "SimpleLinearModel": SimpleLinearModel(),
        "FullyConnectedModel": MLP(),
        "Conv1LayerModel": Conv1LayerModel(),
        "Conv2LayerModel": Conv2LayerModel(),
        "GoodMNISTModel": GoodMNISTModel(),
    }

    model_results = {}

    for model_name, model in tqdm(models.items(), position=0):
        print(f"Training {model_name}...")
        model_acc = train_model(
            model, train_loader, test_loader, epochs=epochs, verbose=verbose
        )
        model_results[model_name] = model_acc

    return model_results


if __name__ == "__main__":
    res = main(True)
    print(res)
