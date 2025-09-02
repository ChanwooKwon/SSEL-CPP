import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

metrics = ['Accuracy', 'MCC', 'ROC-AUC']
before = [0.7888, 0.5778, 0.8715]
after = [0.7856, 0.5716, 0.8662]
p_values = [0.7210, 0.7284, 0.0529]


def get_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


data = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': before + after,
    'Condition': ['Before'] * 3 + ['After'] * 3
})


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=data, x='Metric', y='Score', hue='Condition', palette='Paired')


for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='edge', padding=2, fontsize=12, color='black')


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
plt.savefig("C:/Users/LG_LAB/Desktop/SSELCPP/Method4/compare no_filter/boxplot/ERT_filtering_comparison_5fold.png", dpi=300)
plt.show()
