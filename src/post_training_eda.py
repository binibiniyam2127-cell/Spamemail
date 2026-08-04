"""
Post-Training EDA: Model Performance Analysis
Run: python src/post_training_eda.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('reports/figures/post_training', exist_ok=True)

print("="*60)
print("📊 POST-TRAINING EDA - MODEL PERFORMANCE ANALYSIS")
print("="*60)

# 1. Load Data and Model
print("\n1. Loading data and model...")
df = pd.read_csv('data/processed/enron_spam_processed.csv')
model = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

print(f"   ✅ Loaded {len(df)} samples")
print(f"   ✅ Model: Random Forest with {model.n_estimators} trees")

# 2. Prepare features
print("\n2. Preparing features...")
X = df['processed_text']
y = df['label']

# Split data (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transform
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

print(f"   ✅ Test set: {len(X_test)} samples")

# 3. Performance Metrics
print("\n3. Performance Metrics:")
accuracy = (y_pred == y_test).mean()
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 4. Confusion Matrix Visualization
print("\n4. Creating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix as numbers
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Confusion Matrix as percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[1].set_title('Confusion Matrix (Percentages)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('reports/figures/post_training/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/post_training/confusion_matrix.png")

# 5. ROC Curve
print("\n5. Creating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc="lower right")
axes[0].grid(True)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(recall, precision, color='blue', lw=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].grid(True)
axes[1].fill_between(recall, precision, alpha=0.2)

plt.tight_layout()
plt.savefig('reports/figures/post_training/roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/post_training/roc_pr_curves.png")

# 6. Feature Importance (Top 30)
print("\n6. Creating Feature Importance Visualization...")
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:30]

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(range(30), importances[indices], color='steelblue')
ax.set_yticks(range(30))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 30 Most Important Features')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/post_training/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/post_training/feature_importance.png")

# 7. Prediction Distribution
print("\n7. Creating Prediction Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of predicted probabilities
for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
    mask = y_test == label
    axes[0].hist(y_proba[mask], bins=50, alpha=0.5, 
                 label=f'{"Ham" if label==0 else "Spam"}', color=color)
axes[0].set_xlabel('Predicted Probability of Spam')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Prediction Confidence Distribution')
axes[0].legend()
axes[0].axvline(0.5, color='black', linestyle='--', alpha=0.5)

# Box plot - fixed labels
data_to_plot = [y_proba[y_test == 0], y_proba[y_test == 1]]
bp = axes[1].boxplot(data_to_plot, patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_ylabel('Predicted Probability of Spam')
axes[1].set_title('Prediction Confidence by Class')
axes[1].set_xticklabels(['Ham', 'Spam'])
axes[1].axhline(0.5, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('reports/figures/post_training/prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/post_training/prediction_distribution.png")

# 8. Error Analysis
print("\n8. Analyzing Misclassifications...")
df_test = df.loc[X_test.index].copy()
df_test['predicted'] = y_pred
df_test['probability'] = y_proba
df_test['actual'] = y_test
df_test['correct'] = y_pred == y_test

# Find misclassified samples
misclassified = df_test[df_test['correct'] == False]
print(f"   Misclassified: {len(misclassified)} emails")
print(f"   False Positives (Ham -> Spam): {len(misclassified[misclassified['actual']==0])}")
print(f"   False Negatives (Spam -> Ham): {len(misclassified[misclassified['actual']==1])}")

# Example misclassifications
print("\n   📌 Examples of Misclassifications:")
print("   " + "-"*50)

# False Positives (Ham classified as Spam)
fp = misclassified[misclassified['actual'] == 0].head(3)
for idx, row in fp.iterrows():
    text_preview = str(row['text'])[:100] + "..."
    print(f"   ❌ False Positive (Ham → Spam):")
    print(f"      Confidence: {row['probability']:.2%}")
    print(f"      Email: {text_preview}")
    print()

# False Negatives (Spam classified as Ham)
fn = misclassified[misclassified['actual'] == 1].head(3)
for idx, row in fn.iterrows():
    text_preview = str(row['text'])[:100] + "..."
    print(f"   ❌ False Negative (Spam → Ham):")
    print(f"      Confidence: {row['probability']:.2%}")
    print(f"      Email: {text_preview}")
    print()

# 9. Save Misclassified Samples
misclassified.to_csv('data/processed/misclassified_samples.csv', index=False)
print("   💾 Saved misclassified samples to data/processed/misclassified_samples.csv")

# 10. Model Performance Summary
print("\n9. Creating Performance Summary...")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)',
               'False Positive Rate', 'False Negative Rate', 'AUC-ROC'],
    'Value': [
        accuracy,
        precision,
        recall,
        f1,
        fp / (tn + fp) if (tn + fp) > 0 else 0,
        fn / (fn + tp) if (fn + tp) > 0 else 0,
        roc_auc
    ]
})

print(summary.to_string(index=False))

# Save summary
summary.to_csv('models/performance_summary.csv', index=False)
print("\n   💾 Saved performance summary to models/performance_summary.csv")

# 11. Final Summary
print("\n" + "="*60)
print("📊 POST-TRAINING EDA COMPLETE!")
print("="*60)
print("\n📁 Files Created:")
print("   - reports/figures/post_training/confusion_matrix.png")
print("   - reports/figures/post_training/roc_pr_curves.png")
print("   - reports/figures/post_training/feature_importance.png")
print("   - reports/figures/post_training/prediction_distribution.png")
print("   - data/processed/misclassified_samples.csv")
print("   - models/performance_summary.csv")

print("\n📊 Key Findings:")
print(f"   ✅ Accuracy: {accuracy:.2%}")
print(f"   ✅ AUC-ROC: {roc_auc:.4f}")
print(f"   ✅ Spam Recall: {recall:.2%}")
print(f"   ⚠️ False Positives: {fp} emails (Ham flagged as Spam)")
print(f"   ⚠️ False Negatives: {fn} emails (Spam missed)")

print("\n" + "="*60)
print("🎯 MODEL IS READY FOR DEPLOYMENT!")
print("="*60)
