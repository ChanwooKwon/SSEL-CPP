import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix
)
from catboost import CatBoostClassifier
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent

# ── Path Settings ──────────────────────────────────────────────
MODEL_DIR = current_dir / "basemodels" / "results"
FEATURE_DIR = base_dir / "shap_feature_selection" / "feature_optimization"
TEST_DATA_PATH = base_dir / "data" / "test_mordred2dfeature_knn.csv"
RESULT_DIR = current_dir / "test_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_TXT = RESULT_DIR / "SSELCPP_test_result.txt"

# ── Ensemble Combinations ──────────────────────────────────────
ensemble_combinations = {
    "XGB+LGB": ["XGB", "LGB"]
}

# ── Model File Mapping ─────────────────────────────────────────
model_files = {
    "XGB": MODEL_DIR / "xgb_2_lowess0.2_best_model.pkl",
    "LGB": MODEL_DIR / "lgb_2_lowess_best_model.pkl"
}

# ── Feature File Mapping ───────────────────────────────────────
feature_files = {
    "XGB": FEATURE_DIR / "XGB" / "curve" / "optimal_features_lowess0.2_xgb_accuracy_75.csv",
    "LGB": FEATURE_DIR / "LGBM" / "curve" / "optimal_features_lowess_lgb_accuracy_95.csv"
}

# ── Load Test Set ──────────────────────────────────────────────
df_test = pd.read_csv(TEST_DATA_PATH)
X_test_full = df_test.drop(columns=["id"], errors="ignore")
y_true = df_test["id"].apply(lambda x: 1 if "positive" in x.lower() else 0).values

# ── Prediction Function ────────────────────────────────────────
def get_model_prediction(model_name):
    feature_list = pd.read_csv(feature_files[model_name])["Feature"].tolist()
    X_selected = X_test_full[feature_list]

    if model_name == "CB":
        model = CatBoostClassifier()
        model.load_model(model_files[model_name])
    else:
        model = joblib.load(model_files[model_name])

    return model.predict_proba(X_selected)

# ── Apply Ensemble and Save Results ────────────────────────────
with open(RESULT_TXT, "w", encoding="utf-8") as f:
    f.write("🧪 SSELCPP Test Set Ensemble Evaluation\n")
    f.write("=" * 60 + "\n\n")

    for name, model_list in ensemble_combinations.items():
        try:
            probas = [get_model_prediction(m) for m in model_list]
            avg_proba = np.mean(probas, axis=0)
            y_pred = np.argmax(avg_proba, axis=1)
            y_score = avg_proba[:, 1]

            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score)
            mcc = matthews_corrcoef(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            f.write(f"📊 Combination: {name}\n")
            f.write(f"    - Accuracy:     {acc:.4f}\n")
            f.write(f"    - ROC-AUC:      {auc:.4f}\n")
            f.write(f"    - MCC:          {mcc:.4f}\n")
            f.write(f"    - Precision:    {precision:.4f}\n")
            f.write(f"    - Recall:       {recall:.4f}\n")
            f.write(f"    - F1-Score:     {f1:.4f}\n")
            f.write(f"    - Sensitivity:  {sensitivity:.4f}\n")
            f.write(f"    - Specificity:  {specificity:.4f}\n\n")
            
            f.write(f"📈 Confusion Matrix:\n")
            f.write(f"    True Negative:  {tn}\n")
            f.write(f"    False Positive: {fp}\n")
            f.write(f"    False Negative: {fn}\n")
            f.write(f"    True Positive:  {tp}\n\n")

            print(f"✅ {name} evaluation completed successfully")
            
        except Exception as e:
            error_msg = f"❌ Error evaluating {name}: {str(e)}\n\n"
            f.write(error_msg)
            print(error_msg.strip())

print(f"🎉 Results saved → {RESULT_TXT}")

# ── Additional: Save predictions for further analysis ────────────
predictions_df = pd.DataFrame({
    'id': df_test['id'],
    'true_label': y_true
})

for name, model_list in ensemble_combinations.items():
    try:
        probas = [get_model_prediction(m) for m in model_list]
        avg_proba = np.mean(probas, axis=0)
        y_pred = np.argmax(avg_proba, axis=1)
        y_score = avg_proba[:, 1]
        
        predictions_df[f'{name}_prediction'] = y_pred
        predictions_df[f'{name}_probability'] = y_score
        
    except Exception as e:
        print(f"❌ Error saving predictions for {name}: {e}")

predictions_csv = RESULT_DIR / "SSELCPP_test_predictions.csv"
predictions_df.to_csv(predictions_csv, index=False)
print(f"💾 Detailed predictions saved → {predictions_csv}")