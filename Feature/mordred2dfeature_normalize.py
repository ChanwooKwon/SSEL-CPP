import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 파일 경로
input_file_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
output_file_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2d_normalized.csv"

# ── Load dataset ────────────────────────────────────────────
df = pd.read_csv(input_file_path)

# ── Select features (exclude 'id' column if present) ────────
features = df.drop(columns=['id'], errors='ignore')

# ── Apply MinMaxScaler (normalize to 0–1 range) ─────────────
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(features)

# ── Create DataFrame with normalized values ─────────────────
df_normalized = pd.DataFrame(normalized_data, columns=features.columns)

# ── Reattach 'id' column if it exists ──────────────────────
if 'id' in df.columns:
    df_normalized.insert(0, 'id', df['id'])

# ── Save normalized dataset ─────────────────────────────────
df_normalized.to_csv(output_file_path, float_format='%.6f', index=False)

# ── Print confirmation ─────────────────────────────────────
print(f"[✅] Normalized dataset saved → {output_file_path}")