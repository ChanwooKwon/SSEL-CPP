import pandas as pd

# ── File Paths ───────────────────────────────────────
file1 = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/XGB/feature/optimal_features_lowess0.2_accuracy_75.csv"
file2 = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/LGB/feature/optimal_features_lowess_lgb_accuracy_95.csv"

# ── Load CSV Files ───────────────────────────────────
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# ── Assign Ranks ──────────────────────
df1 = df1.copy()
df2 = df2.copy()
df1["Rank1"] = range(1, len(df1) + 1)
df2["Rank2"] = range(1, len(df2) + 1)

# ── Merge Common Features ──────────────────────
merged = pd.merge(df1[["Feature", "Rank1"]], df2[["Feature", "Rank2"]], on="Feature")
merged["AvgRank"] = merged[["Rank1", "Rank2"]].mean(axis=1)

# ── Sort by Average Rank ───────────────────
ranked_common = merged.sort_values(by="AvgRank").reset_index(drop=True)

# ── Output ─────────────────────────────────
print(f"Number of common features: {len(ranked_common)}")
for i, row in ranked_common.iterrows():
    print(f"{i+1}. {row['Feature']} (AvgRank: {row['AvgRank']:.2f})")

# ── Save to CSV (Optional) ─────────────────────────────────
ranked_common.to_csv("common_ranked_features.csv", index=False)
