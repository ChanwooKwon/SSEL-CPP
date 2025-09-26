import pandas as pd
from pathlib import Path

# ── Get current directory and set relative paths ──────────────────────
current_dir = Path(__file__).parent
base_dir = current_dir.parent

# ── Performance metrics ─────────────────────────────────────
metrics = ['CV Accuracy', 'CV MCC', 'CV AUC']
metric_names = ['Accuracy', 'MCC', 'ROC']

# ── Number of top features to evaluate ─────────────────────
feature_counts = [50, 100, 150, 200, 250, 300]

# ── Base input/output paths (relative paths) ───────────────
base_input_path = base_dir / "results" / "combination_results"
base_output_path = base_dir / "results" / "summary"

# ── Ensure output directory exists ─────────────────────────
base_output_path.mkdir(parents=True, exist_ok=True)

# ── Loop over each feature size ─────────────────────────────
for num_features in feature_counts:
    # Input CSV file path
    input_csv = base_input_path / f"4x4_rf_combination_results_top{num_features}.csv"
    print(f"\n▶ Processing: {input_csv}")

    # Load CSV file
    df = pd.read_csv(input_csv)

    # Generate a separate summary file for each metric
    for metric, metric_name in zip(metrics, metric_names):
        # Initialize an empty DataFrame
        summary = pd.DataFrame()

        # Compute performance averages for each source method
        for source_method in df['source_method'].unique():
            # Compute average performance for each target method
            method_summary = df[df['source_method'] == source_method].groupby('target_method').agg(
                Mean=(metric, 'mean')
            ).reset_index()

            # Rename columns (to distinguish by source method)
            method_summary.columns = ['target_method', f'{source_method}_{metric_name}_Mean']

            # Merge results
            if summary.empty:
                summary = method_summary
            else:
                summary = pd.merge(summary, method_summary, on='target_method')

        # Save output file
        output_file = base_output_path / f"rf_combination_{metric_name}_summary_top{num_features}.csv"
        summary.to_csv(output_file, index=False)
        print(f" {metric_name} (Top {num_features}) summary saved: {output_file}")

        # Preview summary
        print(f"\n {metric_name} (Top {num_features}) Summary:")
        print(summary)