import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#  File paths (set as an example)
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
output_path = "C:/Users/LG_LAB/Desktop/SSELCPP/AfterSHAP/tsne/data/train_tsne.png"

#  Load data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x else 0)
    X = df.drop(columns=["id", "Label"], errors="ignore")
    y = df["Label"].values
    return X, y

X, y = load_data(data_path)

#  Scaling (t-SNE is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Run t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

#  Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
            c='blue', label='non-CPP', edgecolor='k', alpha=0.7, s=40)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
            c='red', label='CPP', edgecolor='k', alpha=0.7, s=40)

#  Title and font size
plt.title("t-SNE of Train Data", fontsize=20)

#  Axis tick font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#  Legend font size
plt.legend(fontsize=16)

#  Adjust layout and save
plt.tight_layout()
plt.savefig(output_path, dpi=1000)
plt.show()
