import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    log_loss, precision_score, recall_score, f1_score 
)
from itertools import combinations
from tqdm import tqdm

# ------------------------------
# Settings
# ------------------------------
OOF_DIR = "C:/Users/LG_LAB/Desktop/SSELCPP/AfterSHAP/OOF_Proba"
TRAIN_PATH = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"

# Model list and file mapping
model_info = {
    "XGB": "xgb_2_lowess0.2_oof_proba.npy",
    "LGB": "lgb_2_lowess_oof_proba.npy",
    "GB":  "gb_2_lowess0.3_oof_proba.npy",
    "CB":  "cb_1_lowess_oof_proba.npy",
    "RF":  "rf_1_maxima__oof_proba.npy",
    "ERT": "ert_2_lowess0.3_oof_proba.npy"
}

# ------------------------------
# Load True Labels
# ------------------------------
def load_labels():
    df = pd.read_csv(TRAIN_PATH)
    if "Label" not in df.columns:
        df["Label"] = df["id"].apply(lambda x: 1 if "positive" in str(x).lower() else 0)
    return df["Label"].values

y_true = load_labels()

# ------------------------------
# Ensemble Evaluation Function
# ------------------------------
def evaluate_soft_voting(probas_list, y_true):
    avg_proba = np.mean(probas_list, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    y_score = avg_proba[:, 1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)
    logloss = log_loss(y_true, avg_proba)                        
    precision = precision_score(y_true, y_pred, zero_division=0) 
    recall = recall_score(y_true, y_pred, zero_division=0)       
    f1 = f1_score(y_true, y_pred, zero_division=0)               
    return acc, auc, mcc, logloss, precision, recall, f1

# ------------------------------
# Evaluate All Combinations and Save Results
# ------------------------------
results = []
model_names = list(model_info.keys())

for r in range(2, len(model_names) + 1):
    for combo in tqdm(list(combinations(model_names, r)), desc=f"Evaluating {r}-model combinations"):
        try:
            probas_list = []
            for model in combo:
                file_path = os.path.join(OOF_DIR, model_info[model])
                probas = np.load(file_path)
                probas_list.append(probas)

            acc, auc, mcc, logloss, precision, recall, f1 = evaluate_soft_voting(probas_list, y_true)
            results.append({
                "n_models": r,
                "models": "+".join(combo),
                "accuracy": acc,
                "roc_auc": auc,
                "mcc": mcc,
                "log_loss": logloss,         
                "precision": precision,      
                "recall": recall,            
                "f1_score": f1               
            })
        except Exception as e:
            print(f" Error in {combo}: {e}")
            continue

# ------------------------------
# Save All Results
# ------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("ensemble_train_results.csv", index=False)
print(" Finished evaluating all combinations and saved overall results.")

# ------------------------------
# Save Top 5 Combinations (Accuracy, MCC, ROC-AUC)
# ------------------------------
def save_top_combinations(df, metric, filename):
    top5 = df.sort_values(by=metric, ascending=False).head(5)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Top 5 combinations by {metric.upper()}:\n\n")
        for i, row in top5.iterrows():
            f.write(f"{i+1}. Models: {row['models']}\n")
            f.write(f"   Accuracy   : {row['accuracy']:.4f}\n")
            f.write(f"   ROC-AUC    : {row['roc_auc']:.4f}\n")
            f.write(f"   MCC        : {row['mcc']:.4f}\n")
            f.write(f"   Log Loss   : {row['log_loss']:.4f}\n")
            f.write(f"   Precision  : {row['precision']:.4f}\n")
            f.write(f"   Recall     : {row['recall']:.4f}\n")
            f.write(f"   F1 Score   : {row['f1_score']:.4f}\n\n")
    print(f" Saved top 5 by {metric} → {filename}")


save_top_combinations(results_df, "accuracy", "top5_acc_train.txt")
save_top_combinations(results_df, "mcc", "top5_mcc_train.txt")
save_top_combinations(results_df, "roc_auc", "top5_roc_auc_train.txt")
