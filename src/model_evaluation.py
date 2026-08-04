"""
Advanced Model Evaluation
Run: python src/model_evaluation.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('reports/evaluation', exist_ok=True)

print("="*60)
print("📊 ADVANCED MODEL EVALUATION")
print("="*60)

# Load data and model
print("\n1. Loading data and model...")
df = pd.read_csv('data/processed/enron_spam_processed.csv')
model = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

print(f"   ✅ Loaded {len(df)} samples")
print(f"   ✅ Model: Random Forest with {model.n_estimators} trees")

# Prepare features
X = df['processed_text']
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

# Calculate all metrics
print("\n2. Calculating metrics...")

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision (Spam)': precision_score(y_test, y_pred),
    'Recall (Spam)': recall_score(y_test, y_pred),
    'F1-Score (Spam)': f1_score(y_test, y_pred),
    'Precision (Ham)': precision_score(y_test, y_pred, pos_label=0),
    'Recall (Ham)': recall_score(y_test, y_pred, pos_label=0),
    'F1-Score (Ham)': f1_score(y_test, y_pred, pos_label=0)
}

print("\n📊 Performance Metrics:")
for metric, value in metrics.items():
    print(f"   {metric:20s}: {value:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (Ham):  {tn:6d}")
print(f"   False Positives:       {fp:6d} (Ham → Spam)")
print(f"   False Negatives:       {fn:6d} (Spam → Ham)")
print(f"   True Positives (Spam): {tp:6d}")

# Additional metrics
print(f"\n📊 Additional Metrics:")
print(f"   False Positive Rate:   {fp/(tn+fp):.4f}")
print(f"   False Negative Rate:   {fn/(fn+tp):.4f}")
print(f"   Precision (Spam):      {tp/(tp+fp):.4f}")
print(f"   Recall (Spam):         {tp/(tp+fn):.4f}")

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('reports/evaluation/metrics.csv', index=False)
print("\n   💾 Saved metrics to reports/evaluation/metrics.csv")

# Create visualization dashboard
print("\n3. Creating evaluation visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# Normalized Confusion Matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[0,1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0,1].set_title('Normalized Confusion Matrix')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1,0].set_xlim([0.0, 1.0])
axes[1,0].set_ylim([0.0, 1.05])
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].set_title('ROC Curve')
axes[1,0].legend(loc="lower right")
axes[1,0].grid(True)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
axes[1,1].plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.4f}')
axes[1,1].set_xlabel('Recall')
axes[1,1].set_ylabel('Precision')
axes[1,1].set_title('Precision-Recall Curve')
axes[1,1].legend(loc="lower left")
axes[1,1].grid(True)
axes[1,1].fill_between(recall, precision, alpha=0.2)

plt.tight_layout()
plt.savefig('reports/evaluation/evaluation_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/evaluation/evaluation_dashboard.png")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print("\n📁 Files created:")
print("   - reports/evaluation/metrics.csv")
print("   - reports/evaluation/evaluation_dashboard.png")
