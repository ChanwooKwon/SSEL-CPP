import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Set data path
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"

#  Load CSV file
df = pd.read_csv(data_path)

#  List of descriptors to analyze
features = ['BIC5', 'AATSC0i', 'ATSC2m', 'BCUTc-1h', 'ETA_epsilon_5']

#  Calculate correlation matrix (Pearson)
cor_matrix = df[features].corr(method='pearson')  
#  Visualization settings
plt.figure(figsize=(10, 8))
sns.set(style='whitegrid', font_scale=1.4)

# Draw heatmap
ax = sns.heatmap(
    cor_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 24}
)

#  Adjust title and label font sizes
plt.title("Pearson Correlation Matrix", fontsize=24, pad=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)

#  Adjust colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)

#  Save and show plot
plt.tight_layout()
plt.savefig("correlation_heatmap_top5_features_pearson.png", dpi=1000)
plt.show()
