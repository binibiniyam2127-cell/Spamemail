"""
Visualize extracted features
Run: python src/visualize_features.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports/figures', exist_ok=True)

# Load processed data
df = pd.read_csv('data/processed/enron_spam_processed.csv')
print(f"📊 Loaded {len(df)} samples")

# Features to visualize
features = ['text_length', 'word_count', 'spam_word_count', 'ham_word_count', 'spam_ratio', 'exclamation_count']

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
        subset = df[df['label'] == label][feature]
        axes[i].hist(subset, bins=50, alpha=0.5, label=f'{"Ham" if label==0 else "Spam"}', color=color)
    axes[i].set_title(f'{feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.savefig('reports/figures/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Saved: reports/figures/feature_distributions.png")

# Summary statistics
print("\n📊 Feature Summary Statistics:")
print(df[features + ['label']].groupby('label').mean())

# Correlation with label
print("\n📊 Correlation with Label:")
correlations = df[features + ['label']].corr()['label'].sort_values(ascending=False)
print(correlations)
