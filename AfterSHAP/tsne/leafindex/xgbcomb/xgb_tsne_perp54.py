import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# ───── Paths and Settings ─────
BASE_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP"
MODEL_NAME = "xgb_2_lowess0.2_best_model"
MODEL_PATH = os.path.join(BASE_DIR, "AfterSHAP", "BestModel", "XGB", "Lowess0.2", f"{MODEL_NAME}.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Feature/mordred2dfeature_knn.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "SHAP_select", "XGB/feature/optimal_features_lowess0.2_accuracy_75.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "AfterSHAP/tsne/leafindex/xgbcomb")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───── Load Data ─────
df = pd.read_csv(DATA_PATH)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)
X_full = df.drop(columns=["Label", "id"], errors="ignore")
y = df["Label"].values

# ───── Load Top Features ─────
def load_top_features(feature_path):
    feat_df = pd.read_csv(feature_path)
    return feat_df["Feature"].tolist()

top_features = load_top_features(FEATURE_PATH)
X = X_full[top_features]

# ───── Load Model and Extract Leaf Indices ─────
model = joblib.load(MODEL_PATH)
print("Using model.apply() for leaf index extraction (XGBoost sklearn API detected)")
leaf_indices = model.apply(X)  # (n_samples, n_trees)
n_samples = leaf_indices.shape[0]
leaf_indices_flat = leaf_indices.reshape(n_samples, -1)

encoder = OneHotEncoder(sparse=False)
features = encoder.fit_transform(leaf_indices_flat)

# ───── Run t-SNE (perplexity=54) ─────
perplexity = 54
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
X_tsne = tsne.fit_transform(features)

# ───── Visualization and Save ─────
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], 
            c='blue', label='non-CPP', edgecolor='k', alpha=0.7, s=40)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], 
            c='red', label='CPP', edgecolor='k', alpha=0.7, s=40)

#  Title and Font Size
plt.title("t-SNE of XGBoost", fontsize=20)

#  Axis Tick Font Size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#  Legend Font Size
plt.legend(fontsize=16)
output_path = os.path.join(OUTPUT_DIR, f"xgb_leaf_tsne_final_p{perplexity}.png")
plt.tight_layout()
plt.savefig(output_path, dpi=1000)
plt.close()

# ───── Evaluation Metrics ─────
sil_score = silhouette_score(X_tsne, y)
ch_score = calinski_harabasz_score(X_tsne, y)
db_score = davies_bouldin_score(X_tsne, y)

# Purity (based on median split on the x-axis)
cluster1_ratio = np.mean(y[X_tsne[:, 0] < np.median(X_tsne[:, 0])])
cluster2_ratio = np.mean(y[X_tsne[:, 0] >= np.median(X_tsne[:, 0])])
purity = max(cluster1_ratio, 1 - cluster1_ratio) + max(cluster2_ratio, 1 - cluster2_ratio)
purity /= 2

# LICE Calculation
def compute_lice(X_embedded, labels, k=10):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_embedded)
    _, indices = nn.kneighbors(X_embedded)
    lice_list = []
    for i in range(len(X_embedded)):
        neighbor_labels = labels[indices[i][1:]]
        label_counts = np.bincount(neighbor_labels, minlength=2)
        p = label_counts / np.sum(label_counts)
        lice = 1 - entropy(p, base=2) / np.log2(len(p)) if np.sum(label_counts) > 0 else 0
        lice_list.append(lice)
    return np.mean(lice_list)

lice_score = compute_lice(X_tsne, y)

# ───── Save Results ─────
metrics = {
    "perplexity": perplexity,
    "silhouette_score": sil_score,
    "calinski_harabasz_score": ch_score,
    "davies_bouldin_score": db_score,
    "purity": purity,
    "LICE": lice_score
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"xgb_tsne_evaluation_p{perplexity}.csv"), index=False)
print(f" Results saved for perplexity={perplexity}")
