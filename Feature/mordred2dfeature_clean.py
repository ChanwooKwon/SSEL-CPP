import pandas as pd
import numpy as np
import re

# 파일 경로
filtered_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_filtered.csv"
final_output_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_cleaned.csv"
type_summary_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/type_summary.csv"

# 1. Load the 'id' column separately
df_id = pd.read_csv(filtered_file, usecols=["id"])

# 2. Load all other columns as strings (excluding 'id')
df = pd.read_csv(filtered_file, dtype=str, usecols=lambda col: col != "id")

# 3. Clean values: remove quotes and convert to float if possible
def clean_and_convert(value):
    if isinstance(value, str):
        value = value.strip("'\"")
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

df = df.applymap(clean_and_convert)

# 4. Insert 'id' back as the first column
df.insert(0, "id", df_id)

# 5. Drop columns with less than 10% unique values (excluding 'id')
threshold = 0.10
cols_to_keep = ["id"] + [
    col for col in df.columns if col != "id" and df[col].nunique(dropna=True) / df.shape[0] >= threshold
]
df_final = df[cols_to_keep]

# 6. Save the cleaned dataset
df_final.to_csv(final_output_file, index=False)
print(f"Final preprocessing completed and saved: {final_output_file}")

# 7. Save type summary (column name, dtype, is numeric?)
type_summary = pd.DataFrame({
    "Column Name": df_final.columns,
    "Data Type": df_final.dtypes.values,
    "Is Numeric": df_final.dtypes.apply(lambda x: x in ['int64', 'float64'])
})
type_summary.to_csv(type_summary_file, index=False)
print(f" Column type summary saved: {type_summary_file}")




