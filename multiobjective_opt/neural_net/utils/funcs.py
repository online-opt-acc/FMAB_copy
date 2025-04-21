"""
эксперимент с обучением моделей на простом датасете

в этом эксперименте рассматриваем несколько моделей на датасете mnist

Одни из них простые, и не могут обучиться, другие могут справиться
с поставленной задачей.
Необходимо вовремя распознать лучшую модель и натренировать ее.
"""

from time import time

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


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
