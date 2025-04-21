# multiobjective_opt
code for online multicriteria optimization



# installation
```[bash]
poetry install
```

# running experiments

```
python experiment_code/synthetic/synthetic_experiments.py
```

```
python experiment_code/neural/mab_cifar.py\
        experiment.mab_params.mab_type=UCB\
        experiment.subexp_name=1\
        experiment.mab_params.train_hyperparams.device='cuda:2'
```
