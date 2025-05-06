import os
import re
from pathlib import Path
import ast
import fire
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from multiobjective_opt.utils.utils import get_fig_set_style, savefig

from experiment_code.neural.draw_utils.drawer import plot_vals

MLFLOW_PATH = "mlruns"
SAVEPATH = Path("./exp_results/figures/neural")

# EXP_NAME = "experiments_cycle_100"
# RUN_NAME_ST = "mab_train;SuccessiveHalving"

# N_ARMS = 5

ARM_KEY = "pulled_arm"
DURATION_KEY = 'pull_rew.eval_rez.duration'

METRIC_KEYS = [
        'pulled_arm',

        'pull_rew.confidence_bound',
        'pull_rew.value',

        'pull_rew.eval_rez.loss',
        'pull_rew.eval_rez.duration',
        'pull_rew.eval_rez.accuracy',

        'test_rew.loss',
        'test_rew.duration',
        'test_rew.accuracy'
    ]



def save_figures(figs, save_name):
    assert SAVEPATH.exists(), f"path '{SAVEPATH}' is not exist."
    # exp_results/figures
    for f_name, fig in figs.items():
        path = SAVEPATH / save_name
        if not path.exists():
            os.mkdir(path)
        f_name = f"{f_name}"
        savefig(fig, path, f_name, bbox_inches="tight")


def runs_filter(run, filter_params):
    return run["tags.mlflow.runName"].startswith(filter_params["RUN_NAME_ST"]) and (run['status'] == "FINISHED")



def parse_run_values(f, n_arms, arm_key, duration_key):
    def get_placeholder_for_val():
        return np.zeros((len(f) - n_arms,n_arms))
    
    duration = np.zeros(n_arms, float)
    key_res = {key: get_placeholder_for_val() for key in f[0].keys()}
    
    for i, elem in enumerate(f[:n_arms]):
        # processing init pulls
        print(elem)
        arm = elem[arm_key]
        duration[arm] = elem[duration_key] - (f[i-1][duration_key] if i > 0 else 0)
        for k, v in elem.items():
            key_res[k][0][arm] = v
    
    # fill experiment values
    for i, elem in enumerate(f[n_arms:-1], 1):
        if len(elem) == 0:
            continue
        arm = elem[arm_key]
        duration[arm] += elem[duration_key] - f[ i + n_arms - 2][duration_key]

        for k, v in elem.items():
            key_res[k][i] = key_res[k][i - 1]
            key_res[k][i][arm] = v

    return duration, key_res

def process_run(run_id, client, n_arms):
    run_res = []
    for key in METRIC_KEYS:
        metric_h = client.get_metric_history(run_id, key=key)
        
        T = len(metric_h)
        if len(run_res) < T:
            run_res = [{} for i in range(T)]

        for i, elem in enumerate(metric_h):                
            run_res[i][key] = int(elem.value) if key == ARM_KEY else elem.value
                
    parsed_run = parse_run_values(run_res,n_arms, ARM_KEY, DURATION_KEY)
    return parsed_run, run_res

def get_loss_hist(key, results):
    run_loss_hist = [r[1][key].T for r in results]
    return run_loss_hist


def plot_figures_by_key(alg_names,
                        run_results,
                        f_key, f_name,
                        s_key, s_name,
                        ):
    n = len(alg_names)
    fig, ax, _ = get_fig_set_style(n, shape=(1, 2), figsize=(12, 6))

    plot_vals(ax=ax[0],
              run_hist=get_loss_hist(f_key, run_results),
              ylabel=f_name,
              ylim=(0.08, 0.85),
              model_names=alg_names)

    plot_vals(ax=ax[1],
              run_hist=get_loss_hist(s_key, run_results),
              ylabel=s_name,
              ylim=(0.5, 2.5),
              model_names=alg_names)

    h, legend_ = ax[0].get_legend_handles_labels()


    ####################################
    #  mix positions in legend. specified for this experiment

    pos = [1, 0, 3, 4, 2]
    print(legend_)
    legend_, h = [legend_[p] for p in pos], [h[p] for p in pos]
    fig.legend(
        h,
        legend_,
        ncol=3,
        bbox_to_anchor=(0.0, -0.06, 1, 0.10),
        loc="outside upper left",
        mode="expand",
        borderaxespad=0,
    )
    ####################################
    fig.tight_layout()
    fig.subplots_adjust(bottom=-0.2)
    return fig


def plot_figures(alg_names, run_results):
    figures = {}
    figures["val_fig"] = plot_figures_by_key(alg_names,
                                            run_results,
                                            f_key = "pull_rew.eval_rez.accuracy",
                                            f_name = "Validation accuracy@1",
                                            s_key="pull_rew.eval_rez.loss",
                                            s_name="Validation loss"
                                    )
    figures["test_fig"] = plot_figures_by_key(alg_names,
                                            run_results,
                                            f_key = "test_rew.accuracy",
                                            f_name = "Test accuracy@1",
                                            s_key="test_rew.loss",
                                            s_name="Test loss"
                                        )
    return figures



def plot_regret(runs, N_ARMS):
    agent_names = ['UCB','UCB_2', "SuccessiveHalving"]
    n = len(agent_names)

    fig, ax, _ = get_fig_set_style(n, shape=(1, 2), figsize=(12, 6))

    filter_params = {}

    for i, col_name in enumerate(["pull_rew.eval_rez.loss", "test_rew.loss"]):
        # plot into ax[i]

        for agent_name in agent_names:
            filter_params["RUN_NAME_ST"] = f"mab_train;{agent_name}"
            selected_runs = runs.apply((lambda x: runs_filter(x, filter_params)), axis=1)

            client = mlflow.MlflowClient()

            run_ids = runs[selected_runs]['run_id'].values
            run_res_list = [process_run(run_id, client, N_ARMS) for run_id in run_ids]

            plot_res = np.array([[elem[col_name] for elem in res[1]] for res in run_res_list])

            plot_res = np.cumsum(plot_res, 1)
            mean_loss = plot_res.mean(0)
            low_loss = np.quantile(plot_res, 0.1, 0)
            high_loss = np.quantile(plot_res, 0.9, 0)

            ax[i].plot(mean_loss, label = agent_name)
            
            x = np.arange(len(mean_loss))
            ax[i].fill_between(x, low_loss, high_loss, color="red", alpha=0.1,)
            
            ax[i].legend()
            ax[i].grid()
            ax[i].set_xlabel(r"$\#$ iterations")
            ax[i].set_ylabel(col_name)
        
    return fig


# EXP_NAME = "experiments_cycle_100"
# RUN_NAME_ST = "mab_train;SuccessiveHalving"
# N_ARMS = 5
def main(EXP_NAME: str, RUN_NAME_ST: str, N_ARMS: int):

    mlflow.set_tracking_uri(MLFLOW_PATH)
    
    # get experiments
    experiment = mlflow.get_experiment_by_name(EXP_NAME)
    assert experiment is not None, f"experiment is not found at {MLFLOW_PATH}"
    runs = mlflow.search_runs(experiment.experiment_id)

    filter_params = {}
    filter_params["RUN_NAME_ST"] = RUN_NAME_ST
    selected_runs = runs.apply(lambda x: runs_filter(x, filter_params), axis=1)


    run_ids = runs[selected_runs]['run_id'].values

    # set client and process runs separately
    client = mlflow.MlflowClient()

    # get experiment names
    run = client.get_run(run_ids[0])
    alg_names =  ast.literal_eval(run.data.params["alg_names"])

    assert len(alg_names) == N_ARMS

    run_results = [process_run(run_id, client, N_ARMS)[0] for run_id in run_ids]

    # now plot_values

    figures: dict = plot_figures(alg_names, run_results)
    figures["regret"] = plot_regret(runs, N_ARMS)

    save_figures(figures, f"{EXP_NAME}_{RUN_NAME_ST}")


if __name__ == "__main__":
    fire.Fire(main)

