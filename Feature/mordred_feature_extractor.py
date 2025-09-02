import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors

np.float = float

def mordred2d_feature_extractor(input_file, output_file):
    """
    Extract 2D Mordred descriptors from a CSV containing SMILES.
    - Input CSV must contain 'id' and 'smiles' columns
    - Excludes descriptors that are known to cause NetworkX/eigenvalue issues
    - Handles Inf/NaN values
    - Always preserves 'id' as the first column
    """
    df = pd.read_csv(input_file)
    df = df[['id', 'smiles']].copy()
    df['id'] = df['id'].astype(str)

    mols = [
        Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        for smiles in df['smiles']
    ]

    # Exclude unstable or risky descriptors
    exclude_keywords = ["BaryszMatrix", "BurdenMatrix", "DetourMatrix", "SpAbs", "Eigen"]

    desc_selected = [
        d for d in descriptors.all
        if not any(keyword in str(d) for keyword in exclude_keywords)
    ]

    calc = Calculator(desc_selected, ignore_3D=True)

    # Compute descriptors
    desc_df = calc.pandas(mols, nproc=1)

    # Replace Inf with NaN (keeps numeric dtype consistent)
    desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Insert 'id' as the first column
    desc_df.insert(0, 'id', df['id'].values)

    desc_df.to_csv(output_file, index=False)
    print(f"[✔] Mordred 2D descriptors saved: {output_file}")


def remove_invalid_feature(input_file, output_file, removed_columns_file):
    """
    Remove invalid or uninformative features:
    - Drops columns with a single unique value
    - Drops columns with all NaN values
    - Drops meaningless string-only columns (non-numeric text)
    - Preserves 'id' column at all times
    - Saves the list of removed columns separately
    """
    df = pd.read_csv(input_file)
    if 'id' not in df.columns:
        raise ValueError("Input file must contain an 'id' column.")

    df['id'] = df['id'].astype(str)

    removed_columns = []
    kept_columns = ['id']  # Ensure 'id' is always first

    # Check for string-only columns with no numeric-looking values
    def is_meaningless_string_column(series: pd.Series) -> bool:
        if series.dtype == object:
            all_str = series.apply(lambda x: isinstance(x, str) or pd.isna(x)).all()
            if not all_str:
                return False
            looks_numeric = series.dropna().apply(
                lambda x: str(x).replace('.', '', 1).lstrip('-').isdigit()
            ).any()
            return not looks_numeric
        return False

    # Evaluate all features except 'id'
    feature_cols = [c for c in df.columns if c != 'id']

    for col in feature_cols:
        col_series = df[col]

        if col_series.nunique(dropna=True) <= 1:
            removed_columns.append(col)
            continue
        if col_series.isna().all():
            removed_columns.append(col)
            continue
        if is_meaningless_string_column(col_series):
            removed_columns.append(col)
            continue

        kept_columns.append(col)

    # Keep only valid features, with 'id' at the front
    filtered_df = df[kept_columns].copy()
    filtered_df.to_csv(output_file, index=False)

    # Save removed columns
    pd.DataFrame({"Removed Columns": removed_columns}).to_csv(removed_columns_file, index=False)

    print(f"[✔] Filtered CSV saved: {output_file}")
    print(f"[ℹ️] Number of removed columns: {len(removed_columns)} → Saved: {removed_columns_file}")


# ✅ Main execution
if __name__ == "__main__":
    # Input dataset files
    train_file = "C:/Users/LG_LAB/Desktop/SSELCPP/data/train.csv"
    test_file  = "C:/Users/LG_LAB/Desktop/SSELCPP/data/test.csv"

    # Output: Raw Mordred descriptor files
    mordred2d_train = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature.csv"
    mordred2d_test  = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/Test_mordred2dfeature.csv"

    mordred2d_feature_extractor(train_file, mordred2d_train)
    mordred2d_feature_extractor(test_file,  mordred2d_test)

    # Output: Filtered feature sets (id always preserved)
    filtered_train = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_filtered.csv"
    filtered_test  = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/Test_mordred2dfeature_filtered.csv"

    removed_train = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/removed_columns_summary.csv"
    removed_test  = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/test_removed_columns_summary.csv"

    remove_invalid_feature(mordred2d_train, filtered_train, removed_train)
    remove_invalid_feature(mordred2d_test,  filtered_test,  removed_test)

    print("[✅] All processing steps completed successfully.")
