"""
SSEL-CPP Model Evaluation Script
Evaluate SSELCPP model performance using test.csv SMILES data
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.ssel_cpp import SSELCPP

def extract_true_labels(test_df):
    """Extract true labels from id column"""
    labels = []
    for id_val in test_df['id']:
        if 'positive' in id_val:
            labels.append(1)  # CPP
        elif 'negative' in id_val:
            labels.append(0)  # Non-CPP
        else:
            raise ValueError(f"Unknown label in id: {id_val}")
    return np.array(labels)

def evaluate_model():
    """Evaluate SSELCPP model on test.csv"""
    print("=" * 60)
    print("SSEL-CPP Model Evaluation on Test Set")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    # Try both possible paths
    github_test_path = Path(__file__).parent / "data" / "test.csv"
    original_test_path = Path(__file__).parent.parent.parent / "data" / "test.csv"
    
    if github_test_path.exists():
        test_df = pd.read_csv(github_test_path)
    elif original_test_path.exists():
        test_df = pd.read_csv(original_test_path)
    else:
        raise FileNotFoundError("test.csv not found in either data directory")
    
    print(f"Total test samples: {len(test_df)}")
    
    # Extract true labels
    y_true = extract_true_labels(test_df)
    positive_count = (y_true == 1).sum()
    negative_count = (y_true == 0).sum()
    print(f"Positive samples (CPP): {positive_count}")
    print(f"Negative samples (Non-CPP): {negative_count}")
    
    # Initialize model
    print("\nInitializing SSEL-CPP model...")
    try:
        # Set correct model directory path
        model_dir = Path(__file__).parent / "models"
        model = SSELCPP(model_dir=str(model_dir))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    probabilities = []
    
    for idx, row in test_df.iterrows():
        smiles = row['smiles']
        id_val = row['id']
        
        try:
            result = model.predict_smiles(smiles)
            predictions.append(result['prediction'])
            probabilities.append(result['probability'])
            
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {id_val}: {e}")
            predictions.append(0)  # Default to negative
            probabilities.append(0.0)
    
    y_pred = np.array(predictions)
    y_proba = np.array(probabilities)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"MCC:          {mcc:.4f}")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    
    # Detailed classification report
    print("\n" + "-" * 40)
    print("Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=['Non-CPP', 'CPP']))
    
    # Confusion Matrix
    print("\n" + "-" * 40)
    print("Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    print("Predicted:  Non-CPP  CPP")
    print(f"Non-CPP:      {cm[0,0]:4d}   {cm[0,1]:3d}")
    print(f"CPP:          {cm[1,0]:4d}   {cm[1,1]:3d}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    
    # Save detailed results
    results_df = test_df.copy()
    results_df['true_label'] = y_true
    results_df['predicted_label'] = y_pred
    results_df['cpp_probability'] = y_proba
    results_df['correct_prediction'] = (y_true == y_pred)
    
    output_file = "test_evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test Set Performance:")
    print(f"- Total samples: {len(test_df)}")
    print(f"- Correct predictions: {(y_true == y_pred).sum()}")
    print(f"- Accuracy: {accuracy*100:.2f}%")
    print(f"- MCC: {mcc:.4f}")
    print(f"- ROC-AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1
    }

if __name__ == "__main__":
    try:
        results = evaluate_model()
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()