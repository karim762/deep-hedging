"""One plotting style for every figure of the project."""
import matplotlib.pyplot as plt

COLORS = {"deep": "#2a78d6", "delta": "#1baf7a", "band": "#eb6834", "other": "#eda100", "grey": "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, grid="y"):
    """Light grid, no top/right spines, muted axes."""
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    return ax


def save(fig, name):
    """Save a figure for the README (assets/ at the repository root)."""
    fig.savefig(f"../assets/{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
