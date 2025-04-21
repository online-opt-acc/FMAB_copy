
from dataclasses import dataclass
from typing import List

import numpy as np
from multiobjective_opt.mab.runner import RUNResult
from multiobjective_opt.neural_net.train import mab_cifar
from multiobjective_opt.synthetic_exp.mab_classes import FuncReward
from multiobjective_opt.utils.utils import get_fig_set_style, LINESTYLES

import matplotlib.pyplot as plt
from multiobjective_opt.utils.utils import get_fig_set_style


def draw_cumulative_regret(results, min_val=0, tight_layout=True):
    rew_hist = np.stack([np.array(r.reward_history) for r in results], 0)
    rew_hist = np.cumsum(rew_hist - min_val, 1)
    rew_mean = rew_hist.mean(0)
    rew_high = np.quantile(rew_hist, 0.95, 0)
    rew_low = np.quantile(rew_hist, 0.05, 0)

    fig, ax, _ = get_fig_set_style(rew_mean.shape[0])
    x = np.arange(rew_mean.shape[0])
    ax.plot(rew_mean)
    ax.fill_between(x, rew_low, rew_high, alpha=0.2,)

    ax.set_xlabel(r"$\#$ iteration, $i$")
    ax.set_ylabel(r"$\text{Regret}, R_O$")
    # ax.grid()
    if tight_layout:
        fig.tight_layout()
    return fig



def draw_all_steps(
    results,
    arms,
    colors=None,
    with_intervals=True,
    xlim=None,
    ylim=None,
    tight_layout=True, ):

    fig, ax, _ = get_fig_set_style(len(arms))
    if colors is None:
        colors = ["b", "g", "r", "black"]
    assert len(colors) >= len(arms)
    values_stacked = np.stack([r.values_history for r in results], 0)
    values_mean = values_stacked.mean(0)
    # values_std = values_stacked.std(0)
    rew_high = np.quantile(values_stacked, 0.95, 0)
    rew_low = np.quantile(values_stacked, 0.05, 0)

    for arm, v_mean, v_low, v_high, color, ls in zip(
        arms, values_mean, rew_low, rew_high, colors, LINESTYLES
    ):
        a_n = arm.name
        x = np.arange(len(v_mean))
        ax.plot(x, v_mean, label=a_n, color=color, linestyle = ls[1])
        if with_intervals:
            ax.fill_between(
                x,
                y1=v_low,
                y2=v_high,
                color=color,
                alpha=0.1,
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\#$ iteration, $i$")
    ax.set_ylabel(r"$f_k(x_i)$")
    plt.legend()
    # ax.grid()
    if tight_layout:
        fig.tight_layout()
    return fig

def draw_deltas(
        results,
        arms,
        min_values,
        colors=None,
        with_intervals=True,
        xlim=None,
        ylim=None,
        tight_layout=True,
    ):
        fig, ax, _ = get_fig_set_style(1)
        if colors is None:
            colors = ["b", "g", "r", "black"]

        values_stacked = np.stack([r.values_history for r in results], 0) - min_values[None, :, None]
        values_mean = values_stacked.mean(0)
        # values_std = values_stacked.std(0)
        
        rew_high = np.quantile(values_stacked, 0.95, 0)
        rew_low = np.quantile(values_stacked, 0.05, 0)

        assert len(colors) >= len(arms)
        for arm, v_mean, v_low, v_high, color, min_val, ls in zip(
            arms, values_mean, rew_low, rew_high, colors, min_values, LINESTYLES
        ):
            a_n = arm.name
            x = np.arange(len(v_mean))
            ax.plot(x, v_mean, label=a_n, color=color, linestyle = ls[1])
            if with_intervals:
                ax.fill_between(
                    x,
                    y1=v_low,
                    y2=v_high,
                    color=color,
                    alpha=0.1,
                )

        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid()
        ax.set_xlabel(r"$\#$ iteration, $i$")
        ax.set_ylabel(r"$f_k(x_i) - f_k^*$")
        
        plt.legend()
        
        if tight_layout:
            fig.tight_layout()
        
        return fig


@dataclass
class PlottableRes:
    values_history: List[List[float]] # results for all arms in all steps
    bounds_history: List[List[float]]
    reward_history: List[float]
    actions_history: List[int]

    @classmethod
    def from_run_res(cls, run_res: RUNResult):
        n_actions = run_res.n_actions
        T = len(run_res.rewards_history)

        arm_values = np.zeros((n_actions,), float)
        arm_bounds = np.zeros((n_actions,), float)

        values_history = [None for _ in range(T-n_actions + 1)]
        bounds_history = [None for _ in range(T-n_actions + 1)]
        reward_history = [None for i in range(T)]

        actions_history = run_res.actions_history

        # get rewards after init step
        for i in range(n_actions):
            reward = run_res.rewards_history[i].value
            bound = run_res.rewards_history[i].confidence_bound
            action = run_res.actions_history[i]
            
            reward_history[i] = reward
            arm_values[action] = reward
            arm_bounds[action] = bound

        values_history[0] = arm_values.copy()
        bounds_history[0] = arm_bounds.copy()

        for i in range(T - n_actions):
            reward = run_res.rewards_history[i + n_actions].value
            bound = run_res.rewards_history[i + n_actions].confidence_bound
            action = run_res.actions_history[i+ n_actions]
            
            reward_history[i + n_actions] = reward
            arm_values[action] = reward
            arm_bounds[action] = bound

            values_history[i + 1] = arm_values.copy()
            bounds_history[i + 1] = arm_bounds.copy()

        values_history = np.array(values_history).T 
        bounds_history = np.array(bounds_history).T  
        return cls(values_history, bounds_history, reward_history, actions_history)


def plot_results(run_results: List[RUNResult], arms, min_vals, ylim = [0., 6.5], colors = None, with_intervals = True):
    # RUNResult -> PlottableRes
    if colors is None:
        colors = ["b", "r", "black", "gray", "yellow"]
        
    alg_res = [PlottableRes.from_run_res(res) for res in run_results]
    print([type(elem) for elem in alg_res])
    figures = {}

    figures["running"] = draw_all_steps( alg_res, arms, colors=colors)
    figures["running_clipped"] = draw_all_steps(alg_res, arms, colors=colors, ylim=ylim, with_intervals=with_intervals)
    figures["cumulative"] = draw_cumulative_regret(alg_res, min(min_vals),)
    figures["deltas"] = draw_deltas(alg_res, arms, min_vals, colors = colors)
    return figures