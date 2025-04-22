

num_runs=10

T=350
exp_name="experiments_cycle"
cuda_device="cuda:2"


#uniform
mab_type="Uniform"
for ((i=0; i<${num_runs};i++))
do
    python experiment_code/neural/mab_cifar.py \
            experiment.name=${exp_name} \
            experiment.subexp_name="${mab_type}-${i}"\
            experiment.mab_params.num_pulls=${T}\
            experiment.mab_params.train_hyperparams.device=${cuda_device}\
            experiment.mab_params.mab_type=${mab_type}
done
