import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
from math import floor, log10

# 🔧 Path Settings
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
stat_path = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/stat_normality_tests_summary.csv"
output_dir = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/Mean_CI_with_MWU/"
os.makedirs(output_dir, exist_ok=True)

#  Load Data
df = pd.read_csv(data_path)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)

#  Load Statistical Summary
stat_df = pd.read_csv(stat_path)
top_features = stat_df["Feature"].tolist()

#  Map Significance Symbols
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

#  Visualization of Mean ± 95% CI + Mann–Whitney U Test
for feature in top_features:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Visualization: Mean + 95% CI
    sns.pointplot(
        data=df, x="Label", y=feature, errorbar="ci", capsize=.1,
        join=False, palette=["skyblue", "salmon"], errwidth=2, ax=ax
    )

    # Mann–Whitney U test
    group0 = df[df["Label"] == 0][feature]
    group1 = df[df["Label"] == 1][feature]
    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
    symbol = get_significance_symbol(p_value)
    p_text = format_p_value_sci_tex(p_value)

    # Add annotation inside the graph (scientific notation)
    ax.annotate(f"{p_text}\n{symbol}",
                xy=(0.5, 0.97), xycoords='axes fraction',
                ha='center', va='top',
                fontsize=11, fontweight='bold')

    # Title and labels
    ax.set_title(f"{feature} (0 = Non-CPP, 1 = CPP)\nMean ± 95% CI", fontsize=12)
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Feature Value")
    ax.grid(True)

    # Save (add _10expressed to filename)
    output_file = os.path.join(output_dir, f"{feature}_mean_CI_MWU_10expressed.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")
