import pandas as pd

# ── Performance metrics ─────────────────────────────────────
metrics = ['CV Accuracy', 'CV MCC', 'CV AUC']
metric_names = ['Accuracy', 'MCC', 'ROC']

# ── Number of top features to evaluate ─────────────────────
feature_counts = [50, 100, 150, 200, 250, 300]

# ── Base input/output paths ─────────────────────────────────
base_input_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/combination_results_perfeaturecount/"
base_output_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Filtermethod/combination_results_perfeaturecount/summary/"

# ── Loop over each feature size ─────────────────────────────
for num_features in feature_counts:
    # Input CSV file path
    input_csv = f"{base_input_path}4x4_rf_combination_results_top{num_features}.csv"
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
        output_file = f"{base_output_path}rf_combination_{metric_name}_summary_top{num_features}.csv"
        summary.to_csv(output_file, index=False)
        print(f"✅ {metric_name} (Top {num_features}) summary saved: {output_file}")

        # Preview summary
        print(f"\n {metric_name} (Top {num_features}) Summary:")
        print(summary)
