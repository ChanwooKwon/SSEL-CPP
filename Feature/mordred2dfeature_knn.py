import pandas as pd
from sklearn.impute import KNNImputer
import joblib

# 📁 File paths
input_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_cleaned.csv"
output_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
imputer_model_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/knn/knn_imputer_model.pkl"

# ✅ 1. Load dataset
df = pd.read_csv(input_file)

# ✅ 2. Drop columns with excessive NaN values (predefined list)
columns_to_drop = [
    "MINsSH", "MAXsSH", "MAXaaaC", "MINaaaC", "MINaaN", "MAXaaN",
    "MINssS", "MAXssS", "MDEN-13", "MDEN-23", "MINsssN", "MAXsssN",
    "MINaaNH", "MAXaaNH", "MINdNH", "MAXdNH",
    "MAXaaCH", "MAXaasC", "MINaasC", "MINaaCH"
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# ✅ 3. Separate 'id' column (KNNImputer only handles numeric features)
df_id = df["id"]
df_features = df.drop(columns=["id"])

# ✅ 4. Fit + apply KNN imputer
imputer = KNNImputer(n_neighbors=5)  # Default distance metric: Euclidean
df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)

# ✅ 5. Save trained imputer model
joblib.dump(imputer, imputer_model_path)
print(f"KNNImputer model saved → {imputer_model_path}")

# ✅ 6. Re-attach 'id' column
df_final = pd.concat([df_id, df_imputed], axis=1)

# ✅ 7. Save final imputed dataset
df_final.to_csv(output_file, index=False)
print(f"Missing values imputed with KNN. Saved file: {output_file}")

# 🔧 Save feature order for reproducibility
pd.DataFrame(df_features.columns.tolist()).to_csv("knn_feature_order.csv", index=False, header=False)
print("Feature order saved to knn_feature_order.csv")
