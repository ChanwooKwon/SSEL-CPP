import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Performance metrics and comparison results
metrics = ['Accuracy', 'MCC', 'ROC-AUC']
before = [0.7837, 0.5681, 0.8696]
after = [0.7824, 0.5658, 0.8633]
p_values = [0.8215, 0.8322, 0.0432]

# Significance symbol function
def get_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# Create DataFrame
data = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': before + after,
    'Condition': ['Before'] * 3 + ['After'] * 3
})

# Style settings
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=data, x='Metric', y='Score', hue='Condition', palette='Paired')

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='edge', padding=2, fontsize=12, color='black')

# Add significance annotations
for i, p in enumerate(p_values):
    x1, x2 = i - 0.2, i + 0.2
    y, h = max(before[i], after[i]) + 0.018, 0.01
    sig = get_significance(p)
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
    plt.text((x1 + x2) / 2, y + h + 0.002,
        f'{sig}\n(p={p:.4f})',
        ha='center', va='bottom', fontsize=12)

plt.ylim(0.5, 0.95)
plt.title('ERT (Before vs After Feature Filtering)')
plt.tight_layout()
plt.savefig("C:/Users/LG_LAB/Desktop/SSELCPP/Allfeatures/compare/barplot/ERT_filtering_comparison_10fold.png", dpi=300)
plt.show()
