"""Rank-correlation analysis: MTEB vs MITE.

This module provides utilities for comparing how models rank on the
standard MTEB leaderboard versus the MITE interaction-task benchmark.
The central question is: *does MTEB performance predict interaction-task
performance?*
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_mteb_scores(path_or_dict: str | Path | dict[str, float]) -> dict[str, float]:
    """Load MTEB leaderboard scores per model.

    Parameters
    ----------
    path_or_dict :
        Either a ``dict`` mapping model names to aggregate MTEB scores,
        or a path to a JSON file with the same structure.

    Returns
    -------
    dict[str, float]
    """
    if isinstance(path_or_dict, dict):
        return {k: float(v) for k, v in path_or_dict.items()}

    path = Path(path_or_dict)
    if not path.exists():
        raise FileNotFoundError(f"MTEB scores file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Support both flat {model: score} and nested {model: {avg: score}} formats
    result: dict[str, float] = {}
    for model, value in data.items():
        if isinstance(value, (int, float)):
            result[model] = float(value)
        elif isinstance(value, dict):
            # Try common keys
            for key in ("avg", "average", "score", "mean"):
                if key in value:
                    result[model] = float(value[key])
                    break
            else:
                # Take the first numeric value
                for v in value.values():
                    if isinstance(v, (int, float)):
                        result[model] = float(v)
                        break
    return result


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def compute_rank_correlation(
    mteb_scores: dict[str, float],
    mite_scores: dict[str, float],
) -> dict[str, Any]:
    """Spearman rank correlation between MTEB and MITE scores.

    Only models present in *both* dictionaries are compared.

    Returns
    -------
    dict with keys:
        ``spearman_rho`` : float
        ``p_value``      : float
        ``n_models``     : int
        ``shared_models``: list[str]
        ``mteb_ranks``   : dict[str, int]   (1-based)
        ``mite_ranks``   : dict[str, int]   (1-based)
    """
    shared = sorted(set(mteb_scores) & set(mite_scores))
    n = len(shared)

    if n < 3:
        return {
            "spearman_rho": 0.0,
            "p_value": 1.0,
            "n_models": n,
            "shared_models": shared,
            "mteb_ranks": {},
            "mite_ranks": {},
        }

    mteb_vals = np.array([mteb_scores[m] for m in shared])
    mite_vals = np.array([mite_scores[m] for m in shared])

    rho, p_value = stats.spearmanr(mteb_vals, mite_vals)

    # Compute ordinal ranks (1 = best / highest score)
    mteb_order = np.argsort(-mteb_vals)
    mite_order = np.argsort(-mite_vals)
    mteb_ranks = {shared[i]: int(rank + 1) for rank, i in enumerate(mteb_order)}
    mite_ranks = {shared[i]: int(rank + 1) for rank, i in enumerate(mite_order)}

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_models": n,
        "shared_models": shared,
        "mteb_ranks": mteb_ranks,
        "mite_ranks": mite_ranks,
    }


# ---------------------------------------------------------------------------
# Formatted comparison table
# ---------------------------------------------------------------------------


def generate_comparison_table(
    mteb_scores: dict[str, float],
    mite_scores: dict[str, float],
    model_names: Sequence[str] | None = None,
) -> str:
    """Return a human-readable table comparing MTEB and MITE rankings.

    Parameters
    ----------
    mteb_scores, mite_scores : per-model scores
    model_names : explicit ordering; defaults to the intersection sorted by
        MITE score descending.

    Returns
    -------
    str  -- formatted table (uses ``tabulate`` if available, falls back to
    plain formatting).
    """
    shared = sorted(set(mteb_scores) & set(mite_scores))
    if model_names is not None:
        shared = [m for m in model_names if m in mteb_scores and m in mite_scores]

    if not shared:
        return "(no shared models to compare)"

    # Sort by MITE score descending by default
    if model_names is None:
        shared.sort(key=lambda m: mite_scores[m], reverse=True)

    # Compute ranks
    corr = compute_rank_correlation(mteb_scores, mite_scores)
    mteb_ranks = corr["mteb_ranks"]
    mite_ranks = corr["mite_ranks"]

    rows = []
    for m in shared:
        rank_delta = mteb_ranks.get(m, 0) - mite_ranks.get(m, 0)
        delta_str = f"+{rank_delta}" if rank_delta > 0 else str(rank_delta)
        rows.append(
            [
                m,
                f"{mteb_scores[m]:.4f}",
                mteb_ranks.get(m, "?"),
                f"{mite_scores[m]:.4f}",
                mite_ranks.get(m, "?"),
                delta_str,
            ]
        )

    headers = ["Model", "MTEB Score", "MTEB Rank", "MITE Score", "MITE Rank", "Rank Delta"]

    try:
        from tabulate import tabulate

        table = tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        # Plain-text fallback
        col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        lines = [fmt.format(*headers), fmt.format(*["-" * w for w in col_widths])]
        for row in rows:
            lines.append(fmt.format(*row))
        table = "\n".join(lines)

    rho = corr["spearman_rho"]
    p = corr["p_value"]
    footer = f"\nSpearman rho = {rho:.4f}  (p = {p:.4e}, n = {corr['n_models']})"
    return table + footer


# ---------------------------------------------------------------------------
# Divergence analysis
# ---------------------------------------------------------------------------


def find_biggest_divergences(
    mteb_ranks: dict[str, int],
    mite_ranks: dict[str, int],
    model_names: Sequence[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find models whose MTEB and MITE ranks diverge the most.

    Parameters
    ----------
    mteb_ranks, mite_ranks : 1-based rank dicts
    model_names : restrict to these models (default: intersection)
    top_k : how many divergences to return

    Returns
    -------
    list of dicts, each with keys ``model``, ``mteb_rank``, ``mite_rank``,
    ``delta`` (MTEB_rank - MITE_rank; positive = model ranks higher on MITE
    than on MTEB), sorted by absolute delta descending.
    """
    if model_names is None:
        shared = sorted(set(mteb_ranks) & set(mite_ranks))
    else:
        shared = [m for m in model_names if m in mteb_ranks and m in mite_ranks]

    divergences = []
    for m in shared:
        delta = mteb_ranks[m] - mite_ranks[m]
        divergences.append(
            {
                "model": m,
                "mteb_rank": mteb_ranks[m],
                "mite_rank": mite_ranks[m],
                "delta": delta,
            }
        )

    divergences.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return divergences[:top_k]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_rank_comparison(
    mteb_ranks: dict[str, int],
    mite_ranks: dict[str, int],
    model_names: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "MTEB vs MITE Rankings",
) -> Any:
    """Scatter plot of MTEB rank vs MITE rank with a diagonal reference line.

    Parameters
    ----------
    mteb_ranks, mite_ranks : 1-based rank dicts
    model_names : restrict to these models
    output_path : if given, save the figure to this path (PNG/PDF/SVG)
    title : plot title

    Returns
    -------
    matplotlib Figure (or None if matplotlib is unavailable)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if model_names is None:
        shared = sorted(set(mteb_ranks) & set(mite_ranks))
    else:
        shared = [m for m in model_names if m in mteb_ranks and m in mite_ranks]

    if not shared:
        return None

    x = [mteb_ranks[m] for m in shared]
    y = [mite_ranks[m] for m in shared]
    n = max(max(x), max(y))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Diagonal (perfect agreement)
    ax.plot([1, n], [1, n], "--", color="gray", linewidth=1, label="Perfect agreement")

    # Points
    ax.scatter(x, y, s=60, zorder=5, edgecolors="black", linewidths=0.5)

    # Labels
    for m, xi, yi in zip(shared, x, y):
        # Shorten long model names for readability
        label = m if len(m) <= 30 else m[:27] + "..."
        ax.annotate(
            label,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("MTEB Rank (1 = best)", fontsize=12)
    ax.set_ylabel("MITE Rank (1 = best)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Invert axes so rank 1 is top-left
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig
