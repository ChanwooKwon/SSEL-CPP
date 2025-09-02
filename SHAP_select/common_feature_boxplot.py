import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

#  Path Settings
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
output_dir = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/Boxplot/"  # Modify to your desired save folder

#  Top 20 Common Features
top20_features = [
    "BIC5",
    "ETA_epsilon_5",
    "AATSC0i",
    "ATSC2m",
    "BCUTc-1h",
    "GATS3c",
    "AMID_N",
    "AXp-5d",
    "AATS7p",
    "ZMIC5",
    "GATS1dv",
    "EState_VSA3",
    "MATS1c",
    "MATS4d",
    "GATS8d",
    "AATSC5s",
    "JGI6",
    "AATSC6c",
    "ATSC1s",
    "GATS2d"
]

# Load Data
df = pd.read_csv(data_path)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)

# Significance Symbol Mapping Function
def get_significance_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# Generate and Save Boxplots for Each Feature
for feature in top20_features:
    plt.figure(figsize=(6, 5))
    sns.boxplot(x="Label", y=feature, data=df, palette=["skyblue", "salmon"])
    
    # Statistical Test
    group0 = df[df["Label"] == 0][feature]
    group1 = df[df["Label"] == 1][feature]
    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
    symbol = get_significance_symbol(p_value)

    # Text Positioning
    ymax = max(group0.max(), group1.max())
    ymin = min(group0.min(), group1.min())
    text_y = ymax + (ymax - ymin) * 0.08
    plt.text(0.5, text_y+0.03, f"p = {p_value:.3e}\n{symbol}",
             ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Title and Axis Settings
    plt.title(f"{feature} (0 = Non-CPP, 1 = CPP)")
    plt.xlabel("Class Label")
    plt.ylabel("Feature Value")
    plt.grid(True)

    # Save File
    output_file = f"{output_dir}{feature}_boxplot.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
