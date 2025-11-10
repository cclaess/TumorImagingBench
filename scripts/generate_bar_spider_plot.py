"""
plot_auc_with_95ci.py

Grouped bar chart (with full asymmetric 95% CI error bars) + radar plot.

Edit:
 - tasks : list of task names (length n_tasks)
 - models : list of model names (length n_models)
 - aucs : shape (n_models, n_tasks) with central AUC estimates (0..1)
 - ci_lower : shape (n_models, n_tasks) with lower bound of 95% CI (0..1)
 - ci_upper : shape (n_models, n_tasks) with upper bound of 95% CI (0..1)

Behavior:
 - Error bars represent full 95% CI: lower error = auc - ci_lower, upper error = ci_upper - auc
 - If any ci_lower >= auc or ci_upper <= auc an informative warning is printed and values are clipped.
 - CI values are clipped to [0,1] for visual safety.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ----------------------------- USER INPUT -----------------------------
tasks = [
    "LUNA16", 
    "DLCS", 
    "NSCLC-Radiomics", 
    "NSCLC-Radiogenomics", 
    "C4KC-KiTS", 
    "Colorectal-Liver-Metastases",
]
models = [
    "SPECTRE (ours)",
    "FMCIB",
    "CT-FM",
    "CT-CLIP",
    "PASTA",
    "VISTA3D",
    "VOCO",
    "SUPREM",
    "Merlin",
    "MedImageInsight",
    "ModelsGenesis",
]

# Central AUC estimates (n_models x n_tasks)
aucs = [
    [0.922, 0.500, 0.500, 0.500, 0.500, 0.500],  # SPECTRE (ours)
    [0.886, 0.676, 0.577, 0.588, 0.687, 0.577],  # FMCIB
    [0.654, 0.592, 0.544, 0.620, 0.463, 0.453],  # CT-FM
    [0.573, 0.494, 0.450, 0.510, 0.493, 0.496],  # CT-CLIP
    [0.665, 0.556, 0.557, 0.570, 0.604, 0.465],  # PASTA
    [0.711, 0.608, 0.583, 0.622, 0.682, 0.488],  # VISTA3D
    [0.494, 0.507, 0.526, 0.461, 0.563, 0.421],  # VOCO
    [0.645, 0.544, 0.561, 0.556, 0.719, 0.482],  # SUPREM
    [0.638, 0.562, 0.570, 0.613, 0.643, 0.431],  # Merlin
    [0.676, 0.585, 0.565, 0.561, 0.562, 0.495],  # MedImageInsight
    [0.806, 0.645, 0.577, 0.610, 0.734, 0.530],  # ModelsGenesis
]

# 95% CI lower and upper bounds (same shape as aucs). Replace with your CI bounds.
ci_lower = [
    [0.913, 0.500, 0.500, 0.500, 0.500, 0.500],  # SPECTRE (ours)
    [0.872, 0.655, 0.550, 0.509, 0.619, 0.509],  # FMCIB
    [0.626, 0.577, 0.522, 0.572, 0.409, 0.395],  # CT-FM
    [0.558, 0.467, 0.426, 0.470, 0.429, 0.417],  # CT-CLIP
    [0.655, 0.532, 0.527, 0.510, 0.539, 0.357],  # PASTA
    [0.692, 0.589, 0.545, 0.567, 0.628, 0.401],  # VISTA3D
    [0.468, 0.487, 0.498, 0.423, 0.476, 0.386],  # VOCO
    [0.619, 0.527, 0.527, 0.500, 0.673, 0.389],  # SUPREM
    [0.625, 0.541, 0.534, 0.579, 0.588, 0.371],  # Merlin
    [0.662, 0.560, 0.516, 0.495, 0.533, 0.430],  # MedImageInsight
    [0.796, 0.624, 0.540, 0.569, 0.671, 0.459],  # ModelsGenesis
]

ci_upper = [
    [0.930, 0.500, 0.500, 0.500, 0.500, 0.500],  # SPECTRE (ours)
    [0.900, 0.696, 0.604, 0.667, 0.754, 0.645],  # FMCIB
    [0.683, 0.607, 0.567, 0.669, 0.518, 0.511],  # CT-FM
    [0.587, 0.522, 0.473, 0.550, 0.556, 0.575],  # CT-CLIP
    [0.675, 0.581, 0.587, 0.630, 0.669, 0.572],  # PASTA
    [0.730, 0.627, 0.620, 0.678, 0.736, 0.575],  # VISTA3D
    [0.519, 0.527, 0.554, 0.499, 0.651, 0.456],  # VOCO
    [0.671, 0.561, 0.594, 0.612, 0.765, 0.575],  # SUPREM
    [0.650, 0.582, 0.605, 0.647, 0.697, 0.492],  # Merlin
    [0.689, 0.609, 0.613, 0.626, 0.590, 0.561],  # MedImageInsight
    [0.816, 0.666, 0.615, 0.650, 0.796, 0.601],  # ModelsGenesis
]

# Options
HIGHLIGHT_MODEL = "SPECTRE (ours)"      # set to model name or None
ERROR_FOR = "all"             # "all" or "highlight_only"
OUTPUT_DIR = "figures"
SAVE_VECTOR = True
SCALE_EACH_AXIS = True      # for radar plot: scale each axis to [0,1] individually
SHOW_ORIGINAL_SCALE = True  # for radar plot: show original scale values on each axis
PAD_FRACTION = 0.08         # for radar plot: fraction of padding beyond min/max per axis
FIG_DPI = 300
ANNOTATE_HIGHLIGHT = True     # annotate highlighted model bars with "AUC (low, high)"
# -------------------------- end USER INPUT -----------------------------

# Convert to numpy arrays and validate shapes
aucs = np.array(aucs, dtype=float)
ci_lower = np.array(ci_lower, dtype=float)
ci_upper = np.array(ci_upper, dtype=float)

n_models = len(models)
n_tasks = len(tasks)
if aucs.shape != (n_models, n_tasks):
    raise ValueError(f"aucs must have shape ({n_models}, {n_tasks}); got {aucs.shape}")
if ci_lower.shape != (n_models, n_tasks):
    raise ValueError(f"ci_lower must have shape ({n_models}, {n_tasks}); got {ci_lower.shape}")
if ci_upper.shape != (n_models, n_tasks):
    raise ValueError(f"ci_upper must have shape ({n_models}, {n_tasks}); got {ci_upper.shape}")

# Clip CI bounds to [0,1] but warn if clipping happens or bounds inconsistent
if np.any(ci_lower < 0) or np.any(ci_upper > 1):
    warnings.warn("Some CI bounds are outside [0,1]. They will be clipped to [0,1] for plotting.")
ci_lower = np.clip(ci_lower, 0.0, 1.0)
ci_upper = np.clip(ci_upper, 0.0, 1.0)

# Ensure ci_lower <= auc <= ci_upper; if not, correct with warnings
bad_lower = np.where(ci_lower > aucs)
bad_upper = np.where(ci_upper < aucs)
if bad_lower[0].size > 0 or bad_upper[0].size > 0:
    warnings.warn("Some CI bounds are inconsistent with central auc values (ci_lower > auc or ci_upper < auc). "
                  "They will be adjusted to match the auc (no visual negative error).")
# Fix: if ci_lower > auc -> set ci_lower = auc; if ci_upper < auc -> set ci_upper = auc
ci_lower = np.minimum(ci_lower, aucs)
ci_upper = np.maximum(ci_upper, aucs)

# Compute asymmetric errors: (lower_errors, upper_errors)
lower_err = aucs - ci_lower
upper_err = ci_upper - aucs
# sanity: ensure non-negative
lower_err = np.maximum(lower_err, 0.0)
upper_err = np.maximum(upper_err, 0.0)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(aucs, index=models, columns=tasks)
df.index.name = "Model"

# -------------------------- Grouped bar chart with asymmetric CI --------------------------
def plot_grouped_bar_with_asymmetric_ci(df, lower_err, upper_err, highlight_model=None, error_for="all",
                                        annotate_highlight=False, output_prefix="grouped_bar_95ci", save_vector=True):
    tasks = df.columns.tolist()
    models = df.index.tolist()
    values = df.values
    n_models = len(models)
    n_tasks = len(tasks)

    fig_w = max(8, 1.2 * n_tasks)
    fig_h = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(n_tasks)
    total_width = 0.85
    single_width = total_width / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * single_width

    for i, model in enumerate(models):
        # determine if we should draw CI for this model
        if error_for == "highlight_only" and highlight_model is not None and model != highlight_model:
            yerr = None
        else:
            yerr = np.vstack((lower_err[i], upper_err[i]))  # shape (2, n_tasks) for asymmetric errorbars

        edge_kw = {}
        lw = 0.7
        if highlight_model is not None and model == highlight_model:
            edge_kw = {"edgecolor": "black"}
            lw = 1.6

        bars = ax.bar(x + offsets[i], values[i], width=single_width, label=model, alpha=0.9, linewidth=lw, **edge_kw)

        if yerr is not None:
            # matplotlib bar accepts yerr as shape (2, N) for asymmetric
            # but bar(..., yerr=...) draws errorbars; we'll use that for alignment and cap support.
            ax.bar(x + offsets[i], values[i], width=single_width, alpha=0.0, yerr=yerr, error_kw={"ecolor":"black","elinewidth":1.0,"capsize":3})

        # annotate highlighted model bars optionally
        if annotate_highlight and highlight_model is not None and model == highlight_model:
            for xi, val, low, up in zip(x + offsets[i], values[i], ci_lower[i], ci_upper[i]):
                txt = f"{val:.2f}\n[{low:.2f},{up:.2f}]"
                ax.text(xi, val + 0.015, txt, ha="center", va="bottom", fontsize="x-small", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("AUC per model for each task (95% CI shown)")
    ncol = 1 if n_models <= 6 else 2
    ax.legend(ncol=ncol, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, f"{output_prefix}.png")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    if save_vector:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close(fig)
    return png_path

# --------------------------- Radar (spider) plot (unchanged) -----------------------
def plot_radar(df,
               highlight_model=None,
               output_prefix="radar_plot",
               save_vector=True,
               scale_each_axis=True,
               show_original_scale=True,
               pad_fraction=0.1,
               ticks_per_spoke=6):
    """
    Radar (spider) plot that optionally normalizes each task axis separately.

    Parameters
    ----------
    df : pandas.DataFrame
        models x tasks (index = models, columns = tasks)
    highlight_model : str or None
        model name to emphasize (thicker line).
    output_prefix : str
        filename prefix for saved figures (PNG/PDF/SVG).
    save_vector : bool
        save PDF and SVG in addition to PNG.
    scale_each_axis : bool
        If True, normalize each task to [0,1] using per-task min/max across models.
        If False, plot raw values on a shared 0..1 radius (original behavior).
    show_original_scale : bool
        If True and scale_each_axis==True, annotate each spoke with the original min/max
        (e.g. "Task A\n[0.72,0.95]") so the reader can interpret normalized radii.
    pad_fraction : float
        fraction of radial range to pad text labels outward for readability.

    Notes
    -----
    - matplotlib's polar axes have a single shared radial axis; this function performs per-axis
      normalization then annotates the spokes with their original ranges so interpretation is preserved.
    """

    tasks = df.columns.tolist()
    models = df.index.tolist()
    values = df.values.astype(float)
    n_tasks = len(tasks)

    # Compute per-task min/max
    task_min = values.min(axis=0)
    task_max = values.max(axis=0)
    # avoid division by zero for constant columns
    task_range = task_max - task_min
    # Apply padding to per-task min/max before normalizing so values don't sit exactly at 0/1
    # pad_fraction is a fraction of the per-task range; if range==0 use a small absolute pad
    small_abs_pad = 0.05
    lows = np.empty_like(task_min)
    highs = np.empty_like(task_max)
    for j in range(n_tasks):
        if task_range[j] <= 0:
            pad = small_abs_pad
        else:
            pad = pad_fraction * task_range[j]
        lows[j] = task_min[j] - pad
        highs[j] = task_max[j] + pad
        # safety: ensure highs > lows
        if highs[j] <= lows[j]:
            highs[j] = lows[j] + small_abs_pad

    # Normalize per-task to [0,1] if requested, using padded lows/highs
    if scale_each_axis:
        norm_vals = np.zeros_like(values)
        for j in range(n_tasks):
            if highs[j] <= lows[j]:
                norm_vals[:, j] = 0.5
            else:
                norm_vals[:, j] = (values[:, j] - lows[j]) / (highs[j] - lows[j])
        plot_vals = np.clip(norm_vals, 0.0, 1.0)
    else:
        # No per-axis scaling; we expect values are already in [0,1]
        plot_vals = values.copy()
        # enforce bounds [0,1]
        plot_vals = np.clip(plot_vals, 0.0, 1.0)

    # angles for each axis and close polygon
    angles = np.linspace(0, 2 * math.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2.0)   # start from top
    ax.set_theta_direction(-1)           # clockwise
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=tasks)
    # remove the shared radial ticks/grid (0..1) because we annotate each spoke separately
    # this avoids the big 0..1 ticks that are not associated with any single task axis
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_ylim(0.0, 1.0)                # normalized radius always 0..1

    # Plot each model using normalized / raw plot_vals but close polygon
    for i, model in enumerate(models):
        vals = plot_vals[i].tolist()
        vals += vals[:1]
        linewidth = 1.0
        alpha = 0.12
        zorder = 1
        if highlight_model is not None and model == highlight_model:
            linewidth = 2.2
            alpha = 0.22
            zorder = 3
        ax.plot(angles, vals, linewidth=linewidth, label=model, zorder=zorder)
        ax.fill(angles, vals, alpha=alpha, zorder=zorder)

    # If we normalized per-task, annotate each spoke with original min/max so readers can interpret
    if scale_each_axis and show_original_scale:
        # compute a radial position slightly beyond the outer 1.0 ring to place labels
        tick_width = 0.03  # angular half-width for small tick marks
        for j, ang in enumerate(angles[:-1]):
            # original and padded ranges for this task
            tmin_orig = task_min[j]
            tmax_orig = task_max[j]
            low_padded = lows[j]
            high_padded = highs[j]
            # draw ticks along the spoke according to the original scale
            if ticks_per_spoke is None or ticks_per_spoke <= 0:
                ticks = np.array([tmin_orig, tmax_orig])
            else:
                ticks = np.linspace(tmin_orig, tmax_orig, ticks_per_spoke)

            # don't draw tiny ticks near the center; define a minimum normalized radius
            min_tick_radius = 0.05
            for k, t in enumerate(ticks):
                # normalized radius for this tick (use padded low/high used for plotting)
                if high_padded > low_padded:
                    r = (t - low_padded) / (high_padded - low_padded)
                else:
                    r = 0.5
                r = float(np.clip(r, 0.0, 1.0))
                # skip ticks that would be plotted too close to the center
                if r < min_tick_radius:
                    continue
                a1 = ang - tick_width
                a2 = ang + tick_width
                # small perpendicular tick mark at radius r
                ax.plot([a1, a2], [r, r], color="#777777", linewidth=0.8, zorder=0)
                # label every tick; minimize font and push slightly outward to avoid overlap
                # place label just outside the tick along the radial direction so it sits next to the tick end
                label_offset = 0.03
                label_r = r + label_offset
                # allow labels slightly beyond 1.0 so they don't overlap the outer ring
                label_r = min(1.0 + 0.1, label_r)
                # center horizontally on the spoke and place above/below depending on angle
                ydir = math.sin(ang)
                ha = "center"
                va = "bottom" if ydir >= 0 else "top"
                ax.text(ang, label_r, f"{t:.2f}", ha=ha, va=va, fontsize="x-small", rotation=0)

    ax.set_title("CT foundation model embeddings", y=1.08)
    # place legend to the right
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05), fontsize="small")

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{output_prefix}.png")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    if save_vector:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_radar_ranks(df, highlight_model=None, output_prefix="radar_rank_plot",
                      save_vector=True, annotate_highlight=True, show_circles=True):
    """Simple spider plot that shows model ranks per task (1 = best = outermost).

    Parameters
    ----------
    df : pandas.DataFrame
        models x tasks (index = models, columns = tasks) with numeric scores (higher is better).
    highlight_model : str or None
        model name to emphasize and optionally annotate with rank numbers.
    output_prefix : str
        filename prefix for saved figures.
    save_vector : bool
        save PDF and SVG in addition to PNG.
    annotate_highlight : bool
        If True and highlight_model provided, annotate that model's rank per spoke.
    show_circles : bool
        If True, draw concentric rings corresponding to rank levels.
    """
    tasks = df.columns.tolist()
    models = df.index.tolist()
    values = df.values.astype(float)
    n_tasks = len(tasks)
    n_models = len(models)

    # Compute ranks per task: 1 = best (highest value)
    # use method='min' so ties get the lowest rank among ties
    ranks_df = df.rank(axis=0, method='min', ascending=False).astype(int)
    max_rank = n_models

    # map rank -> normalized radius in [1/max_rank .. 1.0] with 1 -> 1.0 (outermost)
    norm_ranks = (max_rank - ranks_df + 1) / float(max_rank)

    # angles for each axis and close polygon
    angles = np.linspace(0, 2 * math.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=tasks)
    ax.set_ylim(0.0, 1.0)

    # optionally draw concentric rings for rank levels
    if show_circles:
        thetas = np.linspace(0.0, 2 * math.pi, 360)
        for k in range(1, max_rank + 1):
            r = k / float(max_rank)
            ax.plot(thetas, np.full_like(thetas, r), color="#dddddd", linewidth=0.6, linestyle=(0, (2, 4)), zorder=0)

    # plot each model's rank polygon
    for i, model in enumerate(models):
        vals = norm_ranks.loc[model].values.tolist()
        vals += vals[:1]
        linewidth = 1.0
        alpha = 0.12
        zorder = 1
        if highlight_model is not None and model == highlight_model:
            linewidth = 2.2
            alpha = 0.22
            zorder = 3
        ax.plot(angles, vals, linewidth=linewidth, label=model, zorder=zorder)
        ax.fill(angles, vals, alpha=alpha, zorder=zorder)

    # annotate highlight model with rank numbers at each spoke
    if annotate_highlight and highlight_model is not None and highlight_model in models:
        for j, task in enumerate(tasks):
            rank = int(ranks_df.loc[highlight_model, task])
            r = float(norm_ranks.loc[highlight_model, task])
            label_r = min(1.0 + 0.08, r + 0.04)
            ydir = math.sin(angles[j])
            va = "bottom" if ydir >= 0 else "top"
            ax.text(angles[j], label_r, f"#{rank}", ha="center", va=va, fontsize="x-small")

    ax.set_title("Model ranks per task (1 = best, outside)")
    ncol = 1 if n_models <= 6 else 2
    ax.legend(ncol=ncol, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{output_prefix}.png")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    if save_vector:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close(fig)
    return png_path

# ------------------------------- Main --------------------------------
if __name__ == "__main__":
    print("Input DataFrame (models x tasks):")
    print(df.round(3))
    print()

    gpath = plot_grouped_bar_with_asymmetric_ci(df, lower_err, upper_err,
                                                highlight_model=HIGHLIGHT_MODEL,
                                                error_for=ERROR_FOR,
                                                annotate_highlight=ANNOTATE_HIGHLIGHT,
                                                output_prefix="grouped_bar_95ci",
                                                save_vector=SAVE_VECTOR)
    rpath = plot_radar(df, highlight_model=HIGHLIGHT_MODEL, output_prefix="radar_plot", 
                       save_vector=SAVE_VECTOR, scale_each_axis=SCALE_EACH_AXIS,
                       show_original_scale=SHOW_ORIGINAL_SCALE, pad_fraction=PAD_FRACTION)

    rrpath = plot_radar_ranks(df, highlight_model=HIGHLIGHT_MODEL,
                             output_prefix="radar_rank_plot",
                             save_vector=SAVE_VECTOR,
                             annotate_highlight=ANNOTATE_HIGHLIGHT,
                             show_circles=True)

    print("Saved grouped bar chart (95% CI) to:", gpath)
    print("Saved radar plot to:", rpath)
    if SAVE_VECTOR:
        print("Also saved PDF and SVG versions in", OUTPUT_DIR)
    print("Saved radar rank plot to:", rrpath)