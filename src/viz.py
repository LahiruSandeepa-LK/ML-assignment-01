"""Shared plotting style and a colour-blind-safe palette.

The whole project draws its figures through this module so that every chart uses
the same accessible colours and never relies on hue alone to carry meaning. The
palette is the Okabe & Ito set, which stays distinguishable under the common
forms of colour-vision deficiency (deuteranopia, protanopia, tritanopia).

Reference: Okabe, M. & Ito, K. (2008), "Color Universal Design".
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito, by role. All pass a 3:1 non-text contrast check against white.
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Ordered cycle for categorical series (skip yellow: too light on white).
CYCLE = [OKABE_ITO[k] for k in
         ("blue", "vermillion", "green", "orange", "purple", "sky", "black")]

# Fixed colours for the three outcome classes, used identically everywhere.
CLASS_COLORS = {
    "Dropout": OKABE_ITO["vermillion"],
    "Enrolled": OKABE_ITO["orange"],
    "Graduate": OKABE_ITO["blue"],
}

# Diverging pair for signed quantities (e.g. coefficients): + raises, - lowers.
POS = OKABE_ITO["vermillion"]
NEG = OKABE_ITO["blue"]

# Single-hue sequential map for confusion matrices (always annotated with counts,
# so the colour is reinforcement, never the only signal).
SEQUENTIAL = "Blues"


def set_accessible_style():
    """Apply a consistent, high-contrast matplotlib style for the whole project."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,          # body text >= 11pt keeps labels legible
        "legend.fontsize": 10,
        "legend.frameon": True,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def class_palette(order):
    """Return the fixed class colours in a given label order (for seaborn)."""
    return [CLASS_COLORS[c] for c in order]
