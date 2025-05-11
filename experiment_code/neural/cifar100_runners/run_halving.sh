num_runs=10
cuda_num=$1

for T in 50 100 500
do
    exp_name="experiment_${T}"
    cuda_device="cuda:${cuda_num}"
    mab_type="SuccessiveHalving"
    for ((i=0; i<${num_runs};i++))
    do
        python experiment_code/neural/mab_cifar100.py \
                experiment.name=${exp_name} \
                experiment.subexp_name="${mab_type}-${i}"\
                experiment.mab_params.num_pulls=${T}\
                experiment.mab_params.train_hyperparams.device=${cuda_device}\
                experiment.mab_params.mab_type=${mab_type}
    done
done