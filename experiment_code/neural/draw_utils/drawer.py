import matplotlib.pyplot as plt
import numpy as np

LINESTYLES = [
    ("densely dashed", (0, (3, 1))),
    ("d", "solid"),
    ("d", "dotted"),
    ("d", "dashdot"),
    ("dashdotted", (0, (3, 1, 3, ))),
    ("densely dashed", (0, (5, 1))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("long dash with offset", (1, (1, 0))),
    ("dashed", (0, (5, 5))),
    ("d", "dashed"),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
]

COLORS = ["r", "g", "black", "navy", "brown", "blue", "purple", 'darkgreen',"r", "g", "black", "navy", "brown", "blue", "purple", 'darkgreen',]
MARKERS = ["<", "o", "D", ">", 's',"<", "o", "D", ">", 's',"<", "o", "D", ">", 's']
MARK_EVERY_POS = [19, 18, 25, 20, 22, 19, 18, 25, 20, 22,19, 18, 25, 20, 22]
MARKERSIZE = 10
MARKER_EDGE_WIDTH = 2

# from multiobjective_opt.utils import LINESTYLES, get_fig_set_style
def plot_vals(ax, run_hist, ylabel, ylim, model_names):
    losses = np.stack(run_hist)
    mean_loss = losses.mean(0)
    low_loss = np.quantile(losses, 0.1, 0)
    high_loss = np.quantile(losses, 0.9, 0)

    n_arms = len(mean_loss)

    assert n_arms == len(model_names)
    assert n_arms <= len(MARK_EVERY_POS)
    assert n_arms <= len(MARKERS)
    assert n_arms <= len(COLORS)
    
    for i, (loss, low, high, color, m_name, ls, m, n) in enumerate(zip(mean_loss, 
                                                                     low_loss, 
                                                                     high_loss, 
                                                                     COLORS, 
                                                                     model_names, 
                                                                     LINESTYLES, 
                                                                     MARKERS, 
                                                                     MARK_EVERY_POS)):
        
        ax.plot(loss, label = m_name, color = color, linestyle=ls[1], marker = m, 
                            markersize = MARKERSIZE,
                            markeredgewidth=MARKER_EDGE_WIDTH,
                            markerfacecolor='white',
                            markeredgecolor='black',
                            markevery = n)
        
        # print(m_name, ls[0])

        x = np.arange(len(loss))
        ax.fill_between(x, low, high, color=color, alpha=0.1,)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$\#$ iterations")
    ax.grid()
    # ax.legend()
