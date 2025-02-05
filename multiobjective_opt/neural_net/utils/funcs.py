"""
эксперимент с обучением моделей на простом датасете

в этом эксперименте рассматриваем несколько моделей на датасете mnist

Одни из них простые, и не могут обучиться, другие могут справиться
с поставленной задачей.
Необходимо вовремя распознать лучшую модель и натренировать ее.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .dataset_prepare import ClassificationDatasetsHandlerBase


def train_epoch(model, criterion, optimizer, train_loader, device, max_iter=None):
    model.train()
    running_loss = 0.0

    # iter_num = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss


def eval_model(model, test_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    correct = 0
    total = 0

    loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()

            loss += criterion(outputs, labels).item()

    return {"accuracy": correct / total, "loss": loss / total}


def train_model(
    model, train_loader, test_loader, epochs=10, learning_rate=0.001, verbose=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return_res = {}
    for _ in tqdm(
        range(epochs),
        desc=f"model epochs: {model.__class__.__name__}",
        position=1,
        leave=False,
    ):
        running_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        eval_res = eval_model(model, test_loader, device)

        return_res["running_loss"] = running_loss / len(train_loader)
        return_res.update(eval_res)

        if verbose:
            print(f"epoch_loss: {running_loss}")
            print(f"Test Accuracy: {100 * eval_res['accuracy']:.2f}%")

    return return_res


def estimate_model_coeff(
    model_fabric,
    input_shape,
    num_classes,
    loss_function=None,
    device="cpu",
):
    """
    estimates scale coefficient for confidence bound of model

    there input shape and num_classes specify testing data generation process

    Parameters:

    input_shape: tuple of size (batch_size, num_channels, H, W)
    num_classes: interger, therewill be generated random ints as labels of size (batch_size,)
    """

    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    batch_size = input_shape[0]

    D, G = 0, 0

    for _ in range(10):
        model = model_fabric().to(device)
        params_sum = 0.0
        with torch.no_grad():
            for param in model.parameters():
                params_sum += (param**2).sum().cpu().item()

        D_tmp = 2 * params_sum**0.5

        D = max(D, D_tmp)

        # grad_estimation
        for _ in range(10):
            input = torch.rand(*input_shape) * 2 - 1
            labels = torch.randint(num_classes, (batch_size,))
            output = model(input.to(device)).cpu()
            model.zero_grad()

            loss = loss_function(output, labels)
            loss.backward()

            grad_norm = 0.0
            for _, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm += (
                        (param.grad**2).sum().cpu().item()
                    )  # Норма градиента (L2)

            grad_norm = grad_norm**0.5
            G = max(G, grad_norm)
    return D * G


@dataclass
class ArmUsageStatistics:
    model_name: str
    eval_criterion: Literal["accuracy", "loss"]

    num_pulls: int = 0
    train_steps: int = 0

    train_losses: List[float] = field(default_factory=list)
    eval_results: List[Dict] = field(default_factory=list)


@dataclass
class TrainHyperparameters:
    batch_size: int = 128
    lr: float = 1e-3
    # optimizer: torch.optim.Optimizer = torch.optim.Adam
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArm:
    model: nn.Module
    data_loader: ClassificationDatasetsHandlerBase
    coeff: Union[float, None] = None
    eval_criterion: Literal["accuracy", "loss"] = "loss"
    train_hyperparams: TrainHyperparameters = field(
        default_factory=lambda: TrainHyperparameters()
    )

    def __post_init__(
        self,
    ):
        self.model.to(self.train_hyperparams.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_hyperparams.lr)

        self.train_loader, self.test_loader = self.data_loader.load_dataset(
            self.train_hyperparams.batch_size
        )

        self.epoch_length = self.train_loader.iterator_steps

        self.statistics = ArmUsageStatistics(
            model_name=self.model.__class__.__name__, eval_criterion=self.eval_criterion
        )

        if self.coeff is None:
            self.coeff = self.coeff_estimate()

    def coeff_estimate(self):
        """
        estimate coeff if coeff is not given
        """
        batch_size = self.train_hyperparams.batch_size
        inp_shape = self.data_loader.item_shape
        num_classes = self.data_loader.num_classes
        inp_shape = (batch_size, *inp_shape)

        return estimate_model_coeff(
            self.model.__class__,
            inp_shape,
            num_classes,
            self.criterion,
            self.train_hyperparams.device,
        )

    def pull_epoch(self):
        """
        pull causes one epoch train of model
        """
        data_for_epoch = self.train_loader.get_iterator()
        epoch_loss = train_epoch(
            self.model,
            self.criterion,
            self.optimizer,
            data_for_epoch,
            device=self.train_hyperparams.device,
        )

        self.statistics.train_losses.append(epoch_loss)
        self.statistics.train_steps += self.epoch_length
        self.statistics.num_pulls += 1

    def eval_arm(self):
        """
        evaluate arm to get its function confidence_bound
        """
        eval_res = eval_model(self.model, self.test_loader, self.train_hyperparams.device)
        res_mean = eval_res[self.eval_criterion]

        ucb_val = res_mean - self.coeff / (self.statistics.train_steps**0.5)

        self.statistics.eval_results.append(eval_res)
        return {self.eval_criterion: res_mean, "ucb_val": ucb_val}

    def init_arm(self):
        """
        if there will be an init steps for arm

        for example epoch with lr=0 for statistics computation
        """
        self.pull_epoch()


class UCB_nets:
    def __init__(
        self,
        models: List[nn.Module],
        data_loader: ClassificationDatasetsHandlerBase,
        coeffs: List[float] = None,
        eval_criterion: Literal["accuracy", "loss"] = "loss",
        train_hyperparams: TrainHyperparameters = None,
    ):
        # TODO: добавить оценщик коэффициентов
        assert coeffs is not None, "Coefficietn evaluator is not implemented yet"

        if train_hyperparams is None:
            train_hyperparams = TrainHyperparameters()

        self.eval_criterion = eval_criterion
        self.model_arms: List[ModelArm] = [
            ModelArm(
                model,
                data_loader,
                coeff,
                eval_criterion=eval_criterion,
                train_hyperparams=train_hyperparams,
            )
            for model, coeff in zip(models, coeffs)
        ]

    def ucb_train_models(self, sum_epochs=10, verbose=False):
        n_arms = len(self.model_arms)

        losses = []

        for arm in tqdm(self.model_arms, desc="init steps"):
            arm.init_arm()

        arm_ucb_values = np.array(
            [arm.eval_arm()["ucb_val"] for arm in tqdm(self.model_arms, desc="Init eval")]
        )

        remind_pulls = sum_epochs - n_arms
        for _ in tqdm(range(remind_pulls), desc="Total pulls: "):
            arm_to_pull = np.argmin(arm_ucb_values)
            arm = self.model_arms[arm_to_pull]
            arm.pull_epoch()

            arm_eval_res = arm.eval_arm()

            arm_ucb_values[arm_to_pull] = arm_eval_res["ucb_val"]
            losses.append(arm_eval_res[self.eval_criterion])

        return self.model_arms, losses
