

num_runs=10

T=50
exp_name="experiments_cycle_50"
cuda_device="cuda:2"

mab_type="Hyperband"
for ((i=0; i<${num_runs};i++))
do
    python experiment_code/neural/mab_cifar.py \
            experiment.name=${exp_name} \
            experiment.subexp_name="${mab_type}-${i}"\
            experiment.mab_params.num_pulls=${T}\
            experiment.mab_params.train_hyperparams.device=${cuda_device}\
            experiment.mab_params.mab_type=${mab_type}\
            experiment.mab_params.hyperband_eta=2
done

python experiment_code/neural/draw_utils/plot_experiments.py --EXP_NAME="${exp_name}" --RUN_NAME_ST="mab_train;${mab_type}" --N_ARMS=5