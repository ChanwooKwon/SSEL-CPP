import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
from math import floor, log10
import matplotlib.ticker as mticker
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent

# ── CSV file paths (relative paths) ────────────────────────────────────
file_paths = [
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top50.csv",
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top100.csv",
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top150.csv",
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top200.csv",
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top250.csv",
    base_dir / "results" / "combination_results" / "4x4_rf_combination_results_top300.csv"
]

# ── Output directory (relative path) ───────────────────────────────────
save_dir = base_dir / "results" / "figures"
save_dir.mkdir(parents=True, exist_ok=True)

# ── Evaluation metrics ───────────────────────────────────────────────
metrics = {
    "CV Accuracy": "Accuracy",
    "CV MCC": "MCC", 
    "CV AUC": "ROC-AUC"
}

# ── Y-axis limits per metric ─────────────────────────────────────────
metric_ylim = {
    "CV Accuracy": (0.70, 0.83),
    "CV MCC": (0.46, 0.59),
    "CV AUC": (0.78, 0.91),
}

# ── Significance marker mapping ──────────────────────────────────────
def significance_marker(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

# ── Convert scientific notation to LaTeX format ──────────────────────
def format_p_value_sci_tex(p):
    if p == 0:
        return r"$p < 1 \times 10^{-300}$"
    exponent = floor(log10(abs(p)))
    base = p / 10**exponent
    return fr"$p = {base:.2f} \times 10^{{{exponent}}}$"

# ── Add significance annotation with horizontal + vertical bars ──────
def add_significance_annotation(ax, x1, x2, y, diff, p_value):
    sig = significance_marker(p_value)
    if sig:
        bars = ax.patches
        bar1_top = bars[x1].get_height()
        bar2_top = bars[x2].get_height()

        # Horizontal line
        ax.plot([x1, x2], [y, y], color='black', linewidth=1)

        # Vertical lines (from bar tops to y-level)
        ax.plot([x1, x1], [bar1_top+0.015, y], color='black', linewidth=1)
        ax.plot([x2, x2], [bar2_top+0.015, y], color='black', linewidth=1)

        # Text annotation (stars only)
        ax.text((x1 + x2) / 2, y + 0.00001, f" {sig}",
                ha='center', va='bottom', fontsize=30, linespacing=0.5)

# ── Main loop ────────────────────────────────────────────────────────
sns.set(style="white")  # disable grid style

for path in file_paths:
    df = pd.read_csv(path)
    feature_count = int(path.name.split("_top")[-1].split(".")[0])

    # Remove underscores from method labels (e.g., Method_4 → Method4)
    df["source_method"] = df["source_method"].str.replace("_", "", regex=False)

    # Baseline method
    base_method = "Method4"

    # Custom palette (underscore-free labels)
    custom_palette = {
        "Method4": "#E15759",
        "Method3": "#4E79A7",
        "Method2": "#F1CE63",
        "Method1": "#59A14F"
    }

    for metric_col, metric_label in metrics.items():
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Ensure grid is disabled
        ax.grid(False)

        # Order bars by mean values (across all df)
        mean_df_all = df.groupby("source_method")[metric_col].mean().reset_index()
        order = mean_df_all.sort_values(metric_col, ascending=False)["source_method"]

        sns.barplot(
            data=mean_df_all,
            x="source_method",
            y=metric_col,
            ax=ax,
            palette=custom_palette,
            order=order,
            hue="source_method",
            dodge=False,
            legend=False
        )

        # Baseline group (mean across seeds)
        base_group = df[df["source_method"] == base_method].groupby("seed")[metric_col].mean()

        # Subset for Top-N features (for labeling & ymax)
        subset = df[df["num_features"] == feature_count]
        mean_df_subset = subset.groupby("source_method")[metric_col].mean().reset_index()
        ymax_current = mean_df_subset[metric_col].max()

        # Paired t-tests + significance annotations
        for method in order:
            if method == base_method:
                continue
            other_group = df[df["source_method"] == method].groupby("seed")[metric_col].mean()
            common_seeds = base_group.index.intersection(other_group.index)
            base_vals = base_group.loc[common_seeds]
            other_vals = other_group.loc[common_seeds]

            if len(base_vals) > 1 and len(other_vals) > 1:
                stat, p = ttest_rel(base_vals.values, other_vals.values)
                x1 = list(order).index(base_method)
                x2 = list(order).index(method)
                diff = base_vals.mean() - other_vals.mean()

                # Offset height for each new line
                offset_level = len(ax.lines)
                y_level = ymax_current + 0.0002 + offset_level * 0.004

                print(f"[INFO] {metric_label} Top{feature_count}: {base_method} vs {method} | p = {p:.3e}, Δ = {diff:.4f}")
                add_significance_annotation(ax, x1, x2, y_level, diff, p)

        # Add mean values above bars (Top-N subset only)
        for _, row in mean_df_subset.iterrows():
            xpos = list(order).index(row["source_method"])
            ax.text(
                xpos,
                row[metric_col] + 0.001,
                f'{row[metric_col]:.4f}',
                ha='center',
                va='bottom',
                fontsize=25
            )

        # Apply y-axis limits
        ymin, ymax_fixed = metric_ylim.get(metric_col, (0, 1))
        ax.set_ylim(ymin, ymax_fixed)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        # Fonts
        title_fontsize = 24
        label_fontsize = 24
        tick_fontsize = 24

        ax.set_title(f"{metric_label} Comparison (Top {feature_count})", fontsize=title_fontsize)
        ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.set_xlabel("Source Method", fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        plt.tight_layout()

        # Save figure into the specified directory
        save_path = save_dir / f"{metric_label}_Top{feature_count}_with_significance.png"
        plt.savefig(save_path, dpi=1000)
        plt.close()
        print(f" Saved: {save_path}\n")