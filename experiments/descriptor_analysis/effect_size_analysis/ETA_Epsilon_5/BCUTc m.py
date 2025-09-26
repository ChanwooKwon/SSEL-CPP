import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
from math import floor, log10

#  Path Settings
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
output_dir = "C:/Users/LG_LAB/Desktop/SSELCPP/Descriptor analysis_effectsize/ETA_Epsilon_5/"
os.makedirs(output_dir, exist_ok=True)

#  Load Data
df = pd.read_csv(data_path)
# Create Label (0=non-CPP, 1=CPP) → map to string
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)
df["Label"] = df["Label"].map({0: "non-CPP", 1: "CPP"})
df["Label"] = pd.Categorical(df["Label"], categories=["non-CPP", "CPP"], ordered=True)

#  Features to Analyze
related_features = ["ETA_epsilon_5"]

#  Significance Symbol Mapping Function
def get_significance_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

#  Convert p-value to Scientific Notation (TeX format)
def format_p_value_sci_tex(p):
    if p == 0:
        return r"$p < 1 \times 10^{-300}$"
    exponent = floor(log10(abs(p)))
    base = p / 10**exponent
    return fr"$p = {base:.2f} \times 10^{{-{abs(exponent)}}}$"

#  Cliff's Delta Calculation (CPP − non-CPP direction)
def cliffs_delta(cpp_vals, noncpp_vals):
    nx, ny = len(cpp_vals), len(noncpp_vals)
    total = 0
    for xi in cpp_vals:
        for yi in noncpp_vals:
            if xi > yi:
                total += 1
            elif xi < yi:
                total -= 1
    return total / (nx * ny)

#  Mean-based Visualization + Mann–Whitney Test + Cliff’s Delta
for feature in related_features:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Mean ± 95% CI Visualization
    sns.pointplot(
        data=df, x="Label", y=feature, errorbar="ci", capsize=.1,
        join=False, palette=["skyblue", "salmon"], errwidth=2, ax=ax
    )

    # Split Groups
    group_noncpp = df[df["Label"] == "non-CPP"][feature].dropna()
    group_cpp = df[df["Label"] == "CPP"][feature].dropna()

    # Statistical Test
    stat, p_value = mannwhitneyu(group_noncpp, group_cpp, alternative='two-sided')

    # Cliff's Delta (CPP − non-CPP direction)
    delta = cliffs_delta(group_cpp.values, group_noncpp.values)

    # Annotation Text
    symbol = get_significance_symbol(p_value)
    delta_text = f"$\\delta$ = {delta:.3f}"
    annotation_text = f"{symbol}\n{delta_text}"

    # Add Annotation to Graph
    ax.annotate(annotation_text,
                xy=(0.5, 0.97), xycoords='axes fraction',
                ha='center', va='top',
                fontsize=18, fontweight='bold')

    ax.set_title(f"{feature}", fontsize=22)
    ax.set_ylabel("Feature Value", fontsize=16)

    #  Remove x-axis label & adjust font size
    ax.set_xlabel(None)
    ax.tick_params(axis='both', labelsize=16)

    ax.grid(False)

    # Save
    output_file = os.path.join(output_dir, f"{feature}_mean_CI_MWU_CliffsDelta_m.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=1000)
    plt.close()
    print(f" Saved: {output_file}")
