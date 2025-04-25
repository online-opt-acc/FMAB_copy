
for exp_name in experiments_cycle_100 experiments_cycle_200 experiments_cycle
do

    for mab_type in SuccessiveHalving UCB
    do
        python experiment_code/neural/draw_utils/plot_experiments.py --EXP_NAME="${exp_name}" --RUN_NAME_ST="mab_train;${mab_type}" --N_ARMS=5
    done
done