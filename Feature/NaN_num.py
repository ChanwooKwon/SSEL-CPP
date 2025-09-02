import pandas as pd

# File
file_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_cleaned.csv"
output_file = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/Nan_num.csv"

# CSV file load
df = pd.read_csv(file_path)
total_rows = df.shape[0]

# NaN number and ratio calculation
nan_info = pd.DataFrame({
    "Column Name": df.columns,
    "NaN Count": df.isna().sum().values,
    "NaN Ratio (%)": (df.isna().sum().values / total_rows * 100).round(2)
})

# order by NAN number
nan_info = nan_info.sort_values(by="NaN Count", ascending=False)

# Save
nan_info.to_csv(output_file, index=False)
print(f"NaN num and ratio saved: {output_file}")
