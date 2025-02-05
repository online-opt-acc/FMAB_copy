from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MABReturn:
    values_history: List[List[float]]
    bounds_history: List[List[float]]
    reward_history: List[float]
    selection_history: List[int]

    def draw_all_steps(
        self, arms, colors=None, with_intervals=True, xlim=None, ylim=None
    ):
        fig, ax = plt.subplots()
        if colors is None:
            colors = ["b", "g", "r", "black"]
        assert len(colors) >= len(arms)
        for arm, p_h, c_b, color in zip(
            arms, self.values_history, self.bounds_history, colors
        ):
            a_n = arm.name
            x = np.arange(len(p_h))
            ax.plot(x, p_h, label=a_n, color=color)
            if with_intervals:
                ax.fill_between(
                    x,
                    y1=p_h - c_b,
                    y2=p_h,
                    color=color,
                    alpha=0.1,
                )

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\#$ Iteration")
        ax.set_ylabel(r"$f(x)$")
        plt.legend()

        return fig

    def draw_cumulative_regret(self, min_val=0):
        rew_hist = np.array(self.reward_history)
        cumulative_regret = np.cumsum(rew_hist - min_val)

        fig, ax = plt.subplots()
        ax.plot(cumulative_regret)
        ax.set_xlabel(r"$\#$ Iteration")
        ax.set_ylabel(r"$f_{i_t}(x_{i_t}) - f^*$")
        return fig


class UCB:
    def __init__(self, arms):
        self.arms = arms

    def run(self, T=100):
        arms = self.arms

        values_history = []
        bounds_history = []

        reward_history = []
        selection_history = []

        best_vals = np.array([a.step() for a in arms])
        for _ in range(T):
            confidence_bounds = np.array([a.bounds() for a in arms])
            lower_bounds = best_vals - confidence_bounds
            arm_num = np.argmin(lower_bounds)
            arm = arms[arm_num]
            new_val = arm.step()
            best_vals[arm_num] = new_val

            selection_history.append(arm_num)
            reward_history.append(new_val)
            values_history.append(best_vals.copy())
            bounds_history.append(confidence_bounds.copy())

        values_history = np.array(values_history).T
        bounds_history = np.array(bounds_history).T

        res = MABReturn(values_history, bounds_history, reward_history, selection_history)
        return res


class Greedy:
    def __init__(self, arms, eps=0.01):
        self.arms = arms
        self.eps = eps

    def run(self, T=100):
        arms = self.arms

        values_history = []
        bounds_history = []

        reward_history = []
        selection_history = []

        best_vals = np.array([a.step() for a in arms])
        for _ in range(T):
            confidence_bounds = np.array([a.bounds() for a in arms])
            arm_num = (
                np.argmin(best_vals)
                if np.random.rand() > self.eps
                else np.random.randint(len(arms))
            )
            arm = arms[arm_num]
            new_val = arm.step()
            best_vals[arm_num] = new_val

            selection_history.append(arm_num)
            reward_history.append(new_val)
            values_history.append(best_vals.copy())
            bounds_history.append(confidence_bounds.copy())

        values_history = np.array(values_history).T
        bounds_history = np.array(bounds_history).T

        res = MABReturn(values_history, bounds_history, reward_history, selection_history)
        return res
