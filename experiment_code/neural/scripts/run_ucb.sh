

num_runs=10

T=100
exp_name="cifar100_100"
cuda_device="cuda:2"

# UCB type
mab_type="UCB"
scaler=2
for ((i=0; i<${num_runs};i++))
do
    python experiment_code/neural/mab_cifar100.py \
            experiment.name=${exp_name} \
            experiment.subexp_name="${mab_type}_${scaler}-${i}"\
            experiment.mab_params.num_pulls=${T}\
            experiment.mab_params.train_hyperparams.device=${cuda_device}\
            experiment.mab_params.mab_type=${mab_type}\
            experiment.mab_params.mean_estimator_params.coeff_scaler=${scaler}
done

python experiment_code/neural/draw_utils/plot_experiments.py --EXP_NAME="${exp_name}" --RUN_NAME_ST="mab_train;${mab_type}" --N_ARMS=5