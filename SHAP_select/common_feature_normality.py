import pandas as pd
from scipy.stats import shapiro, levene

#  File Paths
data_path = "C:/Users/LG_LAB/Desktop/SSELCPP/Feature/mordred2dfeature_knn.csv"
save_path = "C:/Users/LG_LAB/Desktop/SSELCPP/SHAP_select/stat_normality_tests_summary.csv"  # Save path

#  Top 20 Feature List
top20_features = [
    "BIC5", "ETA_epsilon_5", "AATSC0i", "ATSC2m", "BCUTc-1h", "GATS3c", "AMID_N",
    "AXp-5d", "AATS7p", "ZMIC5", "GATS1dv", "EState_VSA3", "MATS1c", "MATS4d",
    "GATS8d", "AATSC5s", "JGI6", "AATSC6c", "ATSC1s", "GATS2d"
]

#  Load Data
df = pd.read_csv(data_path)
df["Label"] = df["id"].apply(lambda x: 1 if "positive" in x.lower() else 0)

#  Results Storage
results = []

#  Perform tests for each feature
for feature in top20_features:
    group0 = df[df["Label"] == 0][feature]
    group1 = df[df["Label"] == 1][feature]

    # Normality Test
    p_norm_0 = shapiro(group0).pvalue
    p_norm_1 = shapiro(group1).pvalue

    # Homogeneity of Variance Test
    p_var_equal = levene(group0, group1).pvalue

    # Evaluation
    normality_ok = (p_norm_0 > 0.05) and (p_norm_1 > 0.05)
    equal_var_ok = p_var_equal > 0.05

    # Recommended Statistical Test
    if normality_ok and equal_var_ok:
        test_type = "Student's t-test"
    elif normality_ok and not equal_var_ok:
        test_type = "Welch's t-test"
    else:
        test_type = "Mann-Whitney U"

    results.append({
        "Feature": feature,
        "Normality_p_NonCPP": p_norm_0,
        "Normality_p_CPP": p_norm_1,
        "Variance_Equal_p": p_var_equal,
        "Normality_OK": normality_ok,
        "Equal_Variance_OK": equal_var_ok,
        "Recommended_Test": test_type
    })

#  Create DataFrame
result_df = pd.DataFrame(results)

#  Save to CSV
result_df.to_csv(save_path, index=False)
print(f"Results saved: {save_path}")

#  Console Output
with pd.option_context('display.float_format', '{:.3e}'.format):
    print(result_df)
