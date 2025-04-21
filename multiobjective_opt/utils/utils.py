from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict

import itertools
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

##################################################################################
# set there style and etc parameters of firuges
##################################################################################
LINESTYLES = [
    ("d", "dashdot"),
    ("d", "solid"),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("dashed", (0, (5, 5))),
    ("d", "dotted"),
    ("d", "dashed"),
    ("long dash with offset", (1, (1, 0))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("densely dashed", (0, (2, 2))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashed", (0, (5, 1))),
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

COLORMAP_NAME = "tab20"
DPI = 400
FIGSIZE = (17, 8)
FONTSIZE = 20

def get_fig_set_style(lines_count, shape=(1, 1), figsize=None, params = None):
    # colors_list = [ "indigo", "blue", "grey", "red", "#0b5509", "pink", "coral", "black", "y", "c", "g"]
    # colors_list = [ "#a0a0a0","#303000","#406080", "#500010","#606030", "#800080", "goldenrod", "goldenrod", "goldenrod"]
    colors_list = [ "indigo", "blue", "#ff81c0", "red", "#0b8809", "#666666", "goldenrod", "goldenrod", "goldenrod"]
    if lines_count > 9:
        colors_list = mpl.colormaps["Paired_r"].colors #[:lines_count]
    if params is None:
        params = {
            "legend.fontsize": 20,
            "lines.markersize": 15,
            "lines.linewidth": 2.3,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.size": 15,
            #  "text.usetex": True
        }
    sns.set_context("paper", rc=params)
    # sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})
    if figsize is None:
        fig, ax = plt.subplots(*shape, dpi=DPI)
    else:
        fig, ax = plt.subplots(*shape, dpi=DPI, figsize=figsize,)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.unicode'] = True
    plt.grid(which="both")
    return fig, ax, colors_list

def savefig(fig, path, name, *args, **kwargs):
    fig.tight_layout()
    # fig.savefig(str(path / f"{name}_image.png"))
    fig.savefig(str(path / f"{name}_image.pdf"), *args, **kwargs)

    # data = np.array(fig.canvas.buffer_rgba())
    # weights = [0.2989, 0.5870, 0.1140]
    # data = np.dot(data[..., :-1], weights)
    # plt.imsave(str(path / f"{name}_image_gray.png"), data, cmap="gray")
    # plt.imsave(str(path / f"{name}_image_gray.pdf"), data, cmap="gray")



def flatten_dataclass(obj: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    flat_dict = {}
    
    if is_dataclass(obj):
        for field in fields(obj):
            field_value = getattr(obj, field.name)
            new_key = f"{parent_key}{sep}{field.name}" if parent_key else field.name
            flat_dict.update(flatten_dataclass(field_value, new_key, sep))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            flat_dict.update(flatten_dataclass(item, new_key, sep))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            flat_dict.update(flatten_dataclass(v, new_key, sep))
    else:
        if parent_key:  # Игнорируем "лишние" ключи для примитивов
            flat_dict[parent_key] = obj
    return flat_dict
            