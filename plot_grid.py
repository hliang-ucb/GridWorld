import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from grid_distance_viz import plot_all_targets, _draw_graph_edges



def plot_target_coverage(
    G,
    alpha,
    size=4,
    boundary_distance=4,
    ax=None,
    near_color="darkviolet",
    use_source=False,
    show_edges=True,
    node_half_width=0.3,
    edge_gap=0.05,
    edge_color="k",
    edge_alpha=1,
    edge_lw=2,
    arrow_color="#B7C6E0"):
        

    n_nodes = size * size

    for s in range(n_nodes):
        r, c = divmod(s, size)
        y = r
        counts = set1_freq[set1_freq.source==s+1]['count'].values[0]

        face = near_color #mcolors.to_hex(cmap(min(d, max_d) / max_d))
        edge, lbl_col = near_color, 'k' #"#D7E3FC"
        
        label = f"{counts:.0f}"

        rect = mpatches.FancyBboxPatch(
            (c - node_half_width, y - node_half_width),
            2 * node_half_width, 2 * node_half_width,
            boxstyle="round,pad=0.04",
            linewidth=0.5,
            edgecolor=edge,
            facecolor=face,
            alpha=np.clip(counts*alpha,0,1),
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(c, y, label, ha="center", va="center",
                fontsize=16, color=lbl_col, fontweight="bold",
                fontfamily="monospace", zorder=3)



    _draw_graph_edges(ax, G, size, node_half_width=node_half_width,
                       edge_gap=edge_gap, edge_color=edge_color,
                       edge_alpha=edge_alpha, edge_lw=edge_lw,
                       arrow_color=arrow_color)

    ax.set_xlim(-0.6, size - 0.4)
    ax.set_ylim(-0.6, size - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_target_subset(
    G,
    set1,
    set2,
    size=4,
    set1_color="darkviolet",
    set2_color="forestgreen",
    node_half_width=0.3,
    edge_gap=0.05,
    edge_color="k",
    edge_alpha=1,
    edge_lw=2,
    ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    n_nodes = size * size

    for s in range(n_nodes):
        r, c = divmod(s, size)
        y = r

        if np.isin(s,set1):
            face, edge, lbl_col = set1_color, set1_color, 'k' 
        if np.isin(s,set2):
            face, edge, lbl_col = set2_color, set2_color, 'k' 
            
        label = f"node {s+1}"


        rect = mpatches.FancyBboxPatch(
            (c - node_half_width, y - node_half_width),
            2 * node_half_width, 2 * node_half_width,
            boxstyle="round,pad=0.04",
            linewidth=0.5,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(c, y, label, ha="center", va="center",
                fontsize=8, color=lbl_col, fontweight="bold",
                fontfamily="monospace", zorder=3)

    _draw_graph_edges(ax, G, size, node_half_width=node_half_width,
                       edge_gap=edge_gap, edge_color=edge_color,
                       edge_alpha=edge_alpha, edge_lw=edge_lw)

    ax.set_xlim(-0.6, size - 0.4)
    ax.set_ylim(-0.6, size - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax