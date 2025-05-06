"""
hyperband realization from https://github.com/zygmuntz/hyperband/tree/master

"""

from typing import List
import mlflow
import numpy as np

from collections import defaultdict
from itertools import count

from random import random
from math import log, ceil
from time import time, ctime

import random


from experiment_code.neural.train_cifar import get_models
from multiobjective_opt.mab.arms import Reward
from multiobjective_opt.neural_net.utils.dataset_prepare import CIFAR10Handler
from multiobjective_opt.utils.utils import flatten_dataclass

from multiobjective_opt.neural_net.mab_classes import EvalRez, NeuralArm, NeuralReward

class ModelSampler:
    def __init__(self, data_loader, eval_criterion, train_hyperparams):
        self.data_loader=data_loader
        self.eval_criterion=eval_criterion
        self.train_hyperparams=train_hyperparams
        self.train_loader, self.val_loader, self.test_loader = data_loader.load_dataset(train_hyperparams.batch_size)
		
        counter = count(1)
        self.model_name2hash = defaultdict(lambda: next(counter))

    def sample_models(self, n):
        """
        sample parameters instance to train
        
        :param n: number of instances to sample
        """
        
        tryes = int(np.ceil(n / 5))
        models_dicts = [get_models() for _ in range(tryes)]
        
        models = []
        
        for ml in models_dicts:
            for mod in ml.items():
                models.append(mod)
                _ = self.model_name2hash[mod[0]] #запоминаем названия
        random.shuffle(models)
        models = models[:n]
        
        model_params = []
        for _ in models:
            model_params.append({})
        
        arms: List[NeuralArm] = [
                    NeuralArm(
                        model=model,
                        name = m_name,
                        model_params_kwargs=m_params,
                        train_loader=self.train_loader,
                        test_loader=self.test_loader,
                        val_loader =self.val_loader,
                        eval_criterion=self.eval_criterion,
                        train_hyperparams=self.train_hyperparams,
                    )
                    for (m_name, model), m_params in zip(models, model_params)
            ]
        return arms
	
    def try_params(self, n_pulls, arm: NeuralArm, steps_from_start) -> NeuralReward:
        # results = []
        for i in range((n_pulls)):
            reward: NeuralReward = arm.pull()
            mlflow.log_metrics(flatten_dataclass({"pull_rew": reward}), step = steps_from_start + i)
            mlflow.log_metric('alg_name', self.model_name2hash[arm.name], step = steps_from_start + i)
        
        test_rez: EvalRez = arm.test()
        mlflow.log_metrics(flatten_dataclass({"pull_rew": test_rez}), step = steps_from_start + n_pulls)
        return {"loss" : reward.value}



class HyperbandRunner:
	
    @staticmethod
    def max_iter_find(B, eta):

        def f(x): return np.log(eta * x)**2 * x

        b_eta = B * np.log(eta) ** 2
        left = 1
        right = 2
        while f(right) < b_eta:
            right *= 2

        while abs(left - right) > 0.01:
            mid = (left + right)/2
            f_mid = f(mid)
            if f_mid < b_eta:
                left = mid
            else:
                right = mid
        return int(np.ceil((left + right)/2))


    def __init__( self,
                    model_sampler: ModelSampler,
                    max_iter = 81,
                    eta = 3,
                    max_budget = None
                ):


        if max_budget is not None:
            # then recalculate max_iter for one model
            max_iter = HyperbandRunner.max_iter_find(max_budget, eta)

        self.model_sampler = model_sampler
        self.get_params = model_sampler.sample_models
        self.try_params = model_sampler.try_params

        self.max_iter = max_iter 	# maximum iterations per configuration
        self.eta = eta			# defines configuration downsampling rate (default = 3)


        
        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int(np.floor(self.logeta( self.max_iter )))
        self.B = ( self.s_max + 1 ) * self.max_iter

        print(self.max_iter, self.eta, max_budget, self.s_max, self.B)


        self.results = []	# list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        self.best_params = None
		

	# can be called multiple times

    def estimate_steps(self,_, skip_last = 0, ):
        steps = 0

        for s in reversed( range( self.s_max + 1 )):

            n = int( ceil( self.B * (self.eta ** s) / self.max_iter / ( s + 1 ) ))
            r = self.max_iter * self.eta ** ( -s )

            T = [1 for i in range(n)]

            for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1                
                n_configs = (n * self.eta ** ( -i ))

                n_iterations = int(np.ceil(r * self.eta ** ( i )))

                print( n_iterations, len(T), n_iterations * len(T))

                steps += n_iterations * len(T)
                T = T[ 0:int(np.floor( n_configs / self.eta ))]

        print(f"total_steps: {steps}")

    def run( self,_, skip_last = 0, dry_run = False ):
        steps = 0

        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = self.get_params(n)
            
            for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = (n * self.eta ** ( -i ))
                n_iterations = int(np.ceil(r * self.eta ** ( i )))
                
                val_losses = []
                # early_stops = []
                
                for t in T:
                    self.counter += 1
                    
                    start_time = time()
                    
                    if dry_run:
                        result = { 'loss': random.random(), 'log_loss': random.random(), 'auc': random.random()}
                    else:
                        result = self.try_params( n_iterations, t , steps)		# <---
                        steps += n_iterations
                        
                    
                    seconds = int( round( time() - start_time ))
                    mlflow.log_metrics({"duration": seconds}, step = steps)
                    
                    loss = result["loss"]

                    val_losses.append(loss)
                    
                    # early_stop = result.get( 'early_stop', False )
                    # early_stops.append( early_stop )
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                        self.best_params = t

                    # results.extend(result)
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort( val_losses )
                T = [ T[i] for i in indices]
                T = T[ 0:int(np.floor(n_configs / self.eta ))]
        
        print(f"total_steps: {steps}")
        mlflow.log_param("model_names_hash", dict(self.model_sampler.model_name2hash))
        return self.results
