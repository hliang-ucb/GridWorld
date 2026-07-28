"""
grid_distance_viz.py

Visualize, for a chosen target node, how far (in graph hops) every other node
is from it — laid out on a SIZE x SIZE grid the same way your existing cell
drawing code does (s -> row, col = divmod(s, SIZE)).

Key ideas
---------
- Distance is computed with nx.shortest_path_length(G, target=target), which
  gives the hop count from every node TO `target`. This is the natural choice
  here ("how far is each node from the target") and it's also the version
  that does the right thing automatically if G is a DiGraph (it walks edges
  backwards from the target instead of you having to call G.reverse()
  yourself). If you actually want distance FROM the target outward instead,
  swap to source=target.
- Cell fill color fades from `near_color` (close to target) to `far_color`
  (far from target), interpolated by hop count. Unreachable nodes (not in the
  shortest-path dict at all) get their own muted color.
- The "coverage boundary" is drawn using the grid's *geometric* adjacency
  (up/down/left/right neighbor cells), not G's edges — i.e. it traces the
  outline of "all cells with distance <= boundary_distance" as a region on
  the physical grid, regardless of how convoluted G's actual connectivity is.
- G's *actual* edges are drawn as lines connecting the *edges* of the squares
  (not passing through their centers/faces), so you can see the real
  connectivity — walls that break a grid edge simply have no line, and any
  shortcut/teleport edge shows up as a line cutting across the grid. Squares
  are drawn smaller than one grid unit so there's visible empty space between
  them for these connector lines to occupy.
- Row 0 (nodes 0..size-1) is at the BOTTOM of the plot and the highest-index
  row is at the TOP — i.e. normal mathematical y-axis orientation, not image
  orientation.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
from matplotlib.colors import to_rgba

def _region_boundary_segments(covered, size, half_width):
    """
    Given a set of node ids (covered) laid out on a size x size grid,
    return line segments tracing the outer boundary of that region using
    geometric (grid) adjacency.
    """
    segments = []
    for s in covered:
        r, c = divmod(s, size)
        y = r
        neighbor_cells = {
            "top":    (r - 1, c),
            "bottom": (r + 1, c),
            "left":   (r, c - 1),
            "right":  (r, c + 1),
        }
        for side, (nr, nc) in neighbor_cells.items():
            in_bounds = 0 <= nr < size and 0 <= nc < size
            neighbor_id = nr * size + nc if in_bounds else None
            if neighbor_id is None or neighbor_id not in covered:
                if side == "top":
                    seg = [(c - half_width, y - half_width), (c + half_width, y - half_width)]
                elif side == "bottom":
                    seg = [(c - half_width, y + half_width), (c + half_width, y + half_width)]
                elif side == "left":
                    seg = [(c - half_width, y - half_width), (c - half_width, y + half_width)]
                else:  # right
                    seg = [(c + half_width, y - half_width), (c + half_width, y + half_width)]
                segments.append(seg)
    return segments


def _boundary_point(cx, cy, dx, dy, half_width, gap):
    """
    Starting at center (cx, cy), walk along direction (dx, dy) until exiting
    a square of half-width `half_width` centered there, plus an extra `gap`.
    Since square sides are axis-aligned, the exit point along any ray is
    where the largest axis component reaches half_width — this is exact,
    not an approximation via a circumscribed circle.
    """
    scale = max(abs(dx), abs(dy))
    if scale == 0:
        return cx, cy
    t = (half_width + gap) / scale
    return cx + dx * t, cy + dy * t


def _draw_graph_edges(ax, G, size, node_half_width=1, edge_gap=0.05,
                       edge_color="#8FA3C7", edge_alpha=0.7,
                       edge_lw=1.3, arrow_color="#B7C6E0", arrow_lw=1.5,
                       arrow_alpha=0.95):
    """
    Overlay G's actual edges as lines connecting the *boundaries* of the
    squares (not their centers), so lines occupy the empty space between
    cells rather than cutting across the cell faces. Undirected edges are
    drawn as plain lines; directed edges get an arrowhead. If a DiGraph has
    edges going both ways between two nodes, each direction is drawn with a
    slight curve so the two arrows don't sit exactly on top of each other.
    Self-loops are skipped (no useful geometric representation here).
    """
    directed = G.is_directed()

    def center(s):
        r, c = divmod(s, size)
        return (c, r)

    def clipped_endpoints(xu, yu, xv, yv):
        dx, dy = xv - xu, yv - yu
        start = _boundary_point(xu, yu, dx, dy, node_half_width, edge_gap)
        end = _boundary_point(xv, yv, -dx, -dy, node_half_width, edge_gap)
        return start, end

    if directed:
        for u, v in G.edges():
            if u == v:
                continue
            xu, yu = center(u)
            xv, yv = center(v)
            start, end = clipped_endpoints(xu, yu, xv, yv)
            reciprocal = G.has_edge(v, u)
            rad = 0.15 if reciprocal else 0.0
            arrow = mpatches.FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=arrow_lw,
                color=arrow_color,
                alpha=arrow_alpha,
                zorder=2.5,
            )
            ax.add_patch(arrow)
    else:
        seen = set()
        for u, v in G.edges():
            if u == v:
                continue
            pair = frozenset((u, v))
            if pair in seen:
                continue
            seen.add(pair)
            xu, yu = center(u)
            xv, yv = center(v)
            start, end = clipped_endpoints(xu, yu, xv, yv)
            ax.plot([start[0], end[0]], [start[1], end[1]], color=edge_color,
                    linewidth=edge_lw, alpha=edge_alpha, zorder=2.5,
                    solid_capstyle="round")


def plot_target_coverage(
    G,
    target,
    size=4,
    boundary_distance=4,
    ax=None,
    near_color="darkviolet",
    far_color="thistle",
    unreachable_color="#12141A",
    target_color="darkviolet",
    target_edge="darkviolet",
    default_edge="darkviolet",
    boundary_color="w",
    use_source=False,
    show_edges=True,
    show_boundary=False,
    node_half_width=0.3,
    edge_gap=0.05,
    edge_color="k",
    edge_alpha=1,
    edge_lw=2,
    arrow_color="#B7C6E0",
):
    """
    Draw one grid panel showing hop-distance-to-`target` for every node.

    use_source=False (default): distance is "node -> target" (shortest_path_length
        with target=target). Correct choice for directed graphs when you mean
        "how far is this node from the target".
    use_source=True: distance is "target -> node" (shortest_path_length with
        source=target). Use this if you want outward distance from the target.

    show_edges=True: overlay G's actual edges (lines, or arrows if G is a
        DiGraph) on top of the cells so you can see the real connectivity.

    node_half_width: half the side length of each square (in the same units
        as node spacing, which is 1.0). Smaller values leave more empty space
        between squares for the connecting edge lines to occupy.
    edge_gap: small additional gap between where a line/arrow stops and the
        square's edge, so lines don't visually touch the squares.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    n_nodes = size * size

    if use_source:
        dist = dict(nx.shortest_path_length(G, source=target))
    else:
        dist = dict(nx.shortest_path_length(G, target=target))

    finite_d = [d for n, d in dist.items() if n != target]
    max_d = max(finite_d) if finite_d else 1
    max_d = max(max_d, 1)  # avoid divide-by-zero

    cmap = mcolors.LinearSegmentedColormap.from_list("fade", [near_color, far_color])

    for s in range(n_nodes):
        r, c = divmod(s, size)
        y = r

        if s == target:
            d=0
            face, edge, lbl_col = target_color, target_edge, 'w' # "#AADDAA"
            label = "GOAL"
        elif s not in dist:
            face, edge, lbl_col = unreachable_color, default_edge, 'k' #"#555F73"
            label = f"{s}\n\u00d7"  # x mark for unreachable
        else:
            d = dist[s]
            face = near_color #mcolors.to_hex(cmap(min(d, max_d) / max_d))
            edge, lbl_col = default_edge, 'k' #"#D7E3FC"
            label = f"{d}"

        rect = mpatches.FancyBboxPatch(
            (c - node_half_width, y - node_half_width),
            2 * node_half_width, 2 * node_half_width,
            boxstyle="round,pad=0.04",
            linewidth=0.5,
            edgecolor=edge,
            facecolor=to_rgba(face, alpha=np.clip(1-1/ 5*d,0,1)),
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(c, y, label, ha="center", va="center",
                fontsize=8, color=lbl_col, fontweight="bold",
                fontfamily="monospace", zorder=3)

    if show_edges:
        _draw_graph_edges(ax, G, size, node_half_width=node_half_width,
                           edge_gap=edge_gap, edge_color=edge_color,
                           edge_alpha=edge_alpha, edge_lw=edge_lw,
                           arrow_color=arrow_color)

    # coverage boundary: all cells within boundary_distance hops (incl. target)
    if show_boundary:
        covered = {s for s, d in dist.items() if d <= boundary_distance}
        covered.add(target)
        segments = _region_boundary_segments(covered, size, node_half_width)
        if segments:
            lc = mcoll.LineCollection(segments, colors=boundary_color, linewidths=3,
                                    zorder=4, capstyle="round")
            ax.add_collection(lc)

    ax.set_xlim(-0.6, size - 0.4)
    ax.set_ylim(-0.6, size - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"target={target+1}",
                 color="k", fontsize=9)
    return ax


def plot_all_targets(G, size=4, boundary_distance=4, ncols=4, panel_size=3.0,
                      bg_color="w", **kwargs):
    """One panel per target node, arranged in a grid of subplots."""
    
    n_nodes = size * size
    nrows = int(np.ceil(n_nodes / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_size * ncols, panel_size * nrows))
    fig.patch.set_facecolor(bg_color)

    axes_grid = np.atleast_2d(axes)

    for target in range(n_nodes):
        row = target // ncols
        col = target % ncols
        ax = axes_grid[nrows - 1 - row, col]   # reverse the row order
        ax.set_facecolor(bg_color)
        plot_target_coverage(G, target, size=size,
                              boundary_distance=boundary_distance, ax=ax, **kwargs)

    for ax in axes_grid[n_nodes:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # ---- demo with a *customized* (non-fully-connected) 4x4 grid ----
    SIZE = 4
    G = nx.grid_2d_graph(SIZE, SIZE)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # knock out a few edges to simulate walls / custom connectivity
    walls = [(1, 5), (5, 9), (6, 10), (2, 3), (11, 15)]
    for u, v in walls:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    # add one "teleport" shortcut edge to make it visually non-grid-like
    G.add_edge(0, 15)

    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor("#0B1220")
    ax1.set_facecolor("#0B1220")
    plot_target_coverage(G, target=6, size=SIZE, boundary_distance=4, ax=ax1)
    fig1.tight_layout()
    fig1.savefig("/home/claude/single_target_demo.png", dpi=150,
                 facecolor=fig1.get_facecolor())

    fig2 = plot_all_targets(G, size=SIZE, boundary_distance=4, ncols=4)
    fig2.savefig("/home/claude/all_targets_demo.png", dpi=150,
                 facecolor=fig2.get_facecolor())

    # ---- directed-graph demo, to show arrow edges ----
    DG = nx.DiGraph()
    DG.add_nodes_from(range(SIZE * SIZE))
    for u, v in G.edges():
        DG.add_edge(u, v)          # keep most connections two-way
        DG.add_edge(v, u)
    DG.remove_edge(6, 2)           # ...except make a couple one-way
    DG.remove_edge(9, 8)

    fig3 = plt.figure(figsize=(5, 5))
    ax3 = fig3.add_subplot(111)
    fig3.patch.set_facecolor("#0B1220")
    ax3.set_facecolor("#0B1220")
    plot_target_coverage(DG, target=6, size=SIZE, boundary_distance=4, ax=ax3)
    fig3.tight_layout()
    fig3.savefig("/home/claude/directed_demo.png", dpi=150,
                 facecolor=fig3.get_facecolor())

    print("saved single_target_demo.png, all_targets_demo.png, directed_demo.png")
