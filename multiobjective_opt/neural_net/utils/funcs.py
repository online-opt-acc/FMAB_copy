"""
эксперимент с обучением моделей на простом датасете

в этом эксперименте рассматриваем несколько моделей на датасете mnist

Одни из них простые, и не могут обучиться, другие могут справиться
с поставленной задачей.
Необходимо вовремя распознать лучшую модель и натренировать ее.
"""

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Literal, Union
# from matplotlib.font_manager import weight_dict
import mlflow

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from multiobjective_opt.neural_net.utils.dataset_prepare import (
            ClassifDatasetHandlerBase,
            LoaderCycleHandler
        )



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

        running_loss += loss.detach().item()

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


class EarlyStopper:
    def __init__(self, patience: int, min_delta=0):
        """
        saves last patiense фсс
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, score) -> None:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def train_model(
    model, train_loader, test_loader, epochs=10, verbose=False, device = None,
    early_stop_epochs = 10, early_stop_delta = 0.01, *args, **kwargs,
):
    early_stopper = EarlyStopper(early_stop_epochs, early_stop_delta)  # акураси меняется не более чем на о.1 10 шагов
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = kwargs["lr"], weight_decay=kwargs["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.05,
    #     patience=2,
    #     threshold=0.0001,
    #     threshold_mode="rel",
    #     cooldown=0,
    #     min_lr=1e-5,
    #     eps=1e-08,
    # )

    return_res = {}

    start_time = time()
    epoch = 0
    for _ in tqdm(
        range(epochs),
        desc=f"model epochs: {model.__class__.__name__}",
        position=1,
        leave=False,
    ):
        epoch += 1
        running_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        scheduler.step((running_loss / len(train_loader)))
        eval_res = eval_model(model, test_loader, device)

        mlflow.log_metrics(eval_res, step=epoch)
        return_res["running_loss"] = running_loss / len(train_loader)
        return_res.update(eval_res)

        if verbose:
            print(f"epoch_loss: {running_loss}")
            print(f"Test Accuracy: {100 * eval_res['accuracy']:.2f}%")

        stop = early_stopper(eval_res["accuracy"])
        if stop:
            break

    end_time = time() - start_time
    return_res["runtime"] = end_time
    return_res["runned_epochs"] = epoch
    return return_res


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
    model_params_kwargs: Dict
    train_loader: LoaderCycleHandler
    val_loader: Dataset
    test_loader: Dataset
    model_name: str =  None
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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",
        #     factor=0.05,
        #     patience=2,
        #     threshold=0.0001,
        #     threshold_mode="rel",
        #     cooldown=0,
        #     min_lr=0,
        #     eps=1e-08,
        # )

        self.epoch_length = self.train_loader.iterator_steps

        if self.model_name is None:
            self.model_name = f"{self.model.__class__.__name__}_{id(self.model)}"

        self.statistics = ArmUsageStatistics(
            model_name=self.model_name, eval_criterion=self.eval_criterion
        )

        self.min_val = float("inf")

        if self.coeff is None:
            self.coeff = self.coeff_estimate(self.model_params_kwargs)
        print(self.coeff)

    def set_coeff(self, coeff_val):
        self.coeff = coeff_val

    def coeff_estimate(self, model_params_kwargs):
        """
        estimate coeff if coeff is not given
        """
        raise NotImplementedError()
        # batch_size = self.train_hyperparams.batch_size
        # inp_shape = self.train_loader.item_shape
        # num_classes = self.train_loader.num_classes
        # inp_shape = (batch_size, *inp_shape)

        # return estimate_model_coeff(
        #     self.model.__class__,
        #     model_params_kwargs,
        #     inp_shape,
        #     num_classes,
        #     self.criterion,
        #     self.train_hyperparams.device,
        # )

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
        self.scheduler.step(epoch_loss/ self.epoch_length)

        self.statistics.train_losses.append(epoch_loss)
        self.statistics.train_steps += self.epoch_length
        self.statistics.num_pulls += 1

    def test_arm(self):
        # evaluate arm on test dataset
        eval_res = eval_model(self.model, self.test_loader, self.train_hyperparams.device)
        return {f"test_{k}" : v for k, v in eval_res.items()}

    def eval_arm(self, with_min = True, with_interval = True):
        """
        evaluate arm to get its function confidence_bound
        """
        eval_res = eval_model(self.model, self.val_loader, self.train_hyperparams.device)
        res_mean = eval_res[self.eval_criterion]

        conf_interval = 0
        ucb_val = 0
        if with_interval:
            conf_interval = 1. / ((self.statistics.num_pulls)**0.5)
            if with_min:
                # тогда используем лучший результат за все время для построения лосса
                self.min_val = min(self.min_val, res_mean)
                ucb_val = self.min_val - self.coeff * conf_interval
            else:
                ucb_val = res_mean - self.coeff * conf_interval

        self.statistics.eval_results.append(eval_res)
        return {**eval_res, "ucb_val": ucb_val, "conf_interval" : conf_interval}

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
        model_params: List[Dict],
        model_names: List[str],
        data_loader: ClassifDatasetHandlerBase,
        coeffs: List[float] = None,
        eval_criterion: Literal["accuracy", "loss"] = "loss",
        train_hyperparams: TrainHyperparameters = None,
    ):
        #TODO: добавить оценщик коэффициентов
        assert coeffs is not None, "Coefficietn evaluator is not implemented yet"

        if train_hyperparams is None:
            train_hyperparams = TrainHyperparameters()
        self.eval_criterion = eval_criterion

        train_loader, val_loader, test_loader = data_loader.load_dataset(train_hyperparams.batch_size)
        self.model_arms: List[ModelArm] = [
            ModelArm(
                model=model,
                model_name = m_name,
                model_params_kwargs=m_params,
                train_loader=train_loader,
                test_loader=test_loader,
                val_loader =val_loader,
                coeff=coeff,
                eval_criterion=eval_criterion,
                train_hyperparams=train_hyperparams,

            )
            for model, coeff, m_params, m_name in zip(models, coeffs, model_params, model_names)
        ]

    def ucb_train_models(self, sum_epochs=10, verbose=False,
                        set_params_from_pull = True):
        n_arms = len(self.model_arms)

        
        losses = []
        arm_eval_hist = []
        start_time = time()


        arm_ucb_values = np.zeros(len(self.model_arms))

        # add report for just inited model
        for i in tqdm(range(len(self.model_arms)), desc="Init eval"):
            arm = self.model_arms[i]
            arm_eval = arm.eval_arm(with_interval=False)
            dct = {"arm":i, "duration": time() - start_time, **arm_eval}
            mlflow.log_metrics(dct, step=i)
            mlflow.log_metrics(arm.test_arm(), step=i)
            arm_eval_hist.append(dct)

        for i in tqdm(range(len(self.model_arms)), desc="Init eval"):
            arm = self.model_arms[i]
            # initial pull
            arm.init_arm()
            arm_eval = arm.eval_arm()

            if set_params_from_pull:
                new_coeff = 4 * arm_eval[self.eval_criterion]
                arm.set_coeff(new_coeff)
                arm_ucb_values[i] = arm_eval[self.eval_criterion] - new_coeff * arm_eval["conf_interval"]
            else:
                arm_ucb_values[i] = arm_eval["ucb_val"]

            dct = {"arm":i, "duration": time() - start_time, **arm_eval}
            mlflow.log_metrics(dct, step=i)
            mlflow.log_metrics(arm.test_arm(), step=i)
            arm_eval_hist.append(dct)



        remind_pulls = sum_epochs - n_arms
        for i in tqdm(range(n_arms, remind_pulls+ n_arms), desc="Total pulls: "):
            arm_to_pull = int(np.argmin(arm_ucb_values))
            arm = self.model_arms[arm_to_pull]
            arm.pull_epoch()

            arm_eval_res = arm.eval_arm()
            dct = {"arm":arm_to_pull, "duration": time() - start_time, **arm_eval_res}
            mlflow.log_metrics(dct, step=i)
            mlflow.log_metrics(arm.test_arm(), step=i)
            arm_eval_hist.append(dct)

            arm_ucb_values[arm_to_pull] = arm_eval_res["ucb_val"]
            losses.append(arm_eval_res[self.eval_criterion])

        return self.model_arms, losses, arm_eval_hist
