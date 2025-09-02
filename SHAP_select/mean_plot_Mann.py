import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

#  Path Settings
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

#  Mean-based Visualization + Mann–Whitney U Test
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

    # Annotation: add text inside the plot (scientific notation)
    ax.annotate(f"p = {p_value:.2e}\n{symbol}",
                xy=(0.5, 0.97), xycoords='axes fraction',
                ha='center', va='top',
                fontsize=11, fontweight='bold')

    # Title and labels
    ax.set_title(f"{feature} (0 = Non-CPP, 1 = CPP)\nMean ± 95% CI", fontsize=12)
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Feature Value")
    ax.grid(True)

    # Save plot
    output_file = f"{output_dir}{feature}_mean_CI_MWU.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f" Saved: {output_file}")
