import pandas as pd
import numpy as np
import joblib

# ── File paths ──────────────────────────────────────────────────────────────────
test_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/Test_mordred2dfeature.csv"
string_clean_output = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/test_mordred2dfeature_string_processed.csv"
final_output_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/test_mordred2dfeature_knn.csv"

# Pre-trained artifacts from training
knn_imputer_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/knn/knn_imputer_model.pkl"
feature_order_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/knn/knn_feature_order.csv"

# ── 1) Load id as Series ───────────────────────────────────────────────────────
df_id = pd.read_csv(test_file, usecols=["id"])["id"]

# ── 2) Load all other columns as strings (exclude 'id') ────────────────────────
df_str = pd.read_csv(test_file, dtype=str, usecols=lambda col: col != "id")

# ── 3) Clean values: strip quotes and convert to float, else NaN ──────────────
def clean_and_convert(value):
    if isinstance(value, str):
        value = value.strip("'\"")
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

df_numeric = df_str.applymap(clean_and_convert)

# Keep a snapshot before imputation (with 'id' reattached)
df_string_clean = df_numeric.copy()
df_string_clean.insert(0, "id", df_id)
df_string_clean.to_csv(string_clean_output, index=False)
print(f"[✅] Saved string-cleaned snapshot → {string_clean_output}")

# ── 4) Enforce training feature order ──────────────────────────────────────────
# Load trained imputer and the exact training feature order
imputer = joblib.load(knn_imputer_path)
feature_order = pd.read_csv(feature_order_path, header=None).iloc[:, 0].tolist()

# Ensure all training features exist in test
missing_in_test = set(feature_order) - set(df_numeric.columns)
if missing_in_test:
    raise ValueError(f"Missing features in test set (expected from training): {sorted(missing_in_test)}")

# Reorder test columns to match training
df_features_ordered = df_numeric[feature_order]

# ── 5) Apply KNN imputation ───────────────────────────────────────────────────
df_imputed = pd.DataFrame(
    imputer.transform(df_features_ordered),
    columns=feature_order
)

# ── 6) Reattach 'id' and save final output ─────────────────────────────────────
df_final = pd.concat([df_id, df_imputed], axis=1)
df_final.to_csv(final_output_file, index=False)
print(f"[✅] KNN-imputed test set saved → {final_output_file}")
