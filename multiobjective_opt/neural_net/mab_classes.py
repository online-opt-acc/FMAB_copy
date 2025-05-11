from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from multiobjective_opt.mab.arms import Reward
from multiobjective_opt.mab.environment import ArmEnv, BaseArm
from multiobjective_opt.mab.reward_estimators import BaseRewardEstimator

from multiobjective_opt.neural_net.utils.dataset_prepare import (
            # ClassificationDatasetsHandlerBase,
            ClassifDatasetHandlerBase,
            LoaderCycleHandler
        )
from multiobjective_opt.neural_net.utils.funcs import eval_model, train_epoch

@dataclass
class EvalRez:
    accuracy: float
    loss: float
    duration: float = 0

class EvalCriterion(Enum):
    ACCURACY: str = "accuracy"
    LOSS: str = "loss"

@dataclass
class NeuralReward(Reward):
    confidence_bound: float
    eval_rez: EvalRez


@dataclass
class ArmUsageStatistics:
    model_name: str
    eval_criterion: EvalCriterion

    num_pulls: int = 0
    train_steps: int = 0
    train_losses: List[float] = field(default_factory=list)
    # eval_results: List[EvalRez] = field(default_factory=list)

@dataclass
class TrainHyperparameters:
    batch_size: int = 128
    lr: float = 1e-3
    scheduler_t_max: int = 500
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer: torch.optim.Optimizer = torch.optim.Adam


@dataclass
class NeuralArm(BaseArm):
    model: nn.Module
    model_params_kwargs: Dict
    train_loader: LoaderCycleHandler
    val_loader: Dataset
    test_loader: Dataset
    name: str =  None
    eval_criterion: EvalCriterion = EvalCriterion.LOSS
    train_hyperparams: TrainHyperparameters = field(
        default_factory=lambda: TrainHyperparameters())

    def __post_init__(self):
        self.model.to(self.train_hyperparams.device)

        # set up optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_hyperparams.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.train_hyperparams.scheduler_t_max
            )

        # set up
        self.epoch_length = self.train_loader.iterator_steps

        if self.name is None:
            self.name = f"{self.model.__class__.__name__}_{id(self.model)}"

        self.statistics = ArmUsageStatistics(
            model_name=self.name, eval_criterion=self.eval_criterion
        )

    def _train_epoch(self):
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


    def _init_arm(self):
        """
        if there will be an init steps for arm

        for example epoch with lr=0 for statistics computation
        """
        self._train_epoch()
    
    def _eval_on_dataset(self, dataloader):
        eval_res = eval_model(self.model, dataloader, self.train_hyperparams.device)
        rez = EvalRez(**eval_res)
        return rez

    def eval(self):
        return self._eval_on_dataset(self.val_loader)

    def test(self,) -> EvalRez:
        return self._eval_on_dataset(self.test_loader)  
    
    def pull(self, eval = True, *args, **kwargs):
        self._train_epoch()

        if eval:
            eval_res = self.eval()
        
            value = eval_res.accuracy \
                if self.eval_criterion == EvalCriterion.ACCURACY else \
                    eval_res.loss


            conf_interval = 1. / ((self.statistics.num_pulls)**0.5)

            rez = NeuralReward(value = value, confidence_bound=conf_interval, eval_rez = eval_res)
            return rez
        else:
            return NeuralReward(0,0,0)

class NeuralRewardEstimator(BaseRewardEstimator):
    def __init__(self, n_actions, conf_on_min= True, c = None, coeff_scaler = 4):
        """
        
        :parameter: conf_on_min -- if true, then use minimal 
        seen value of function as UCB mean value. etc uses last
        observed value

        :parameter: c -- scaling coefficients for confidence bound
        If None, then it will be approximated from the first arm evaluation
        """

        super().__init__(n_actions)
        self.conf_on_min = conf_on_min

        self.estimations = np.zeros((n_actions,), float)
        self._function_values = np.zeros((n_actions,), float)
        self.confidence_bounds = np.ones((n_actions,), float) * np.inf
        self.coeff_scaler = coeff_scaler
        
        if c is None:
            self.c = np.ones((n_actions,), float)
            self.update_on_first_iter = True
        else:
            assert (len(c) == n_actions) and (np.all(c >= 0))
            self.c = c
            self.update_on_first_iter = False

    def get_estimations(self):
        return -(self._function_values - self.c * self.confidence_bounds)

    def update(self, action, reward: NeuralArm):

        if self.conf_on_min and (not np.isinf(self.confidence_bounds[action])):
            self._function_values[action] = min(self._function_values[action], reward.value)
        else:
            self._function_values[action] = reward.value

        if np.isinf(self.confidence_bounds[action]) and self.update_on_first_iter:
            self.c[action] = self.coeff_scaler * reward.value

        self.confidence_bounds[action] = reward.confidence_bound


class NeuralArmEnv(ArmEnv):
    def __init__(self,
        models: List[nn.Module],
        model_params: List[Dict],
        model_names: List[str],
        data_loader: ClassifDatasetHandlerBase,
        eval_criterion: EvalCriterion = EvalCriterion.LOSS,
        train_hyperparams: TrainHyperparameters = None,
        ):
        # prepare functional arms
            if train_hyperparams is None:
                train_hyperparams = TrainHyperparameters()

            train_loader, val_loader, test_loader = data_loader.load_dataset(train_hyperparams.batch_size)
            arms: List[NeuralArm] = [
                NeuralArm(
                    model=model,
                    name = m_name,
                    model_params_kwargs=m_params,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader =val_loader,
                    eval_criterion=eval_criterion,
                    train_hyperparams=train_hyperparams,
                )
                for model, m_params, m_name in zip(models, model_params, model_names)
            ]
        # initialize environment fully
            super().__init__(arms)
    def pull(self, action, eval = True):
        return self.arms[action].pull(eval )
