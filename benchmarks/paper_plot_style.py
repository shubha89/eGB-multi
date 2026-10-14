"""Shared Matplotlib style for manuscript figures."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np


AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 11
TITLE_SIZE = 12
LEGEND_SIZE = 9.5
ANNOTATION_SIZE = 9.0


def use_paper_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": TICK_LABEL_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "savefig.dpi": 240,
        }
    )


def apply_tick_style(axes) -> None:
    for ax in np.ravel(np.asarray(axes, dtype=object)):
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
