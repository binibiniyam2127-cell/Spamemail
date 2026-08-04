"""
Exploratory Data Analysis for Enron Spam Dataset
Run: python src/eda.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os

warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

print("="*60)
print("📊 EXPLORATORY DATA ANALYSIS - ENRON SPAM DATASET")
print("="*60)

# 1. Load Data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/enron_spam_data.csv')
print(f"   ✅ Loaded {len(df)} emails")

# 2. Basic Info
print("\n2. Basic Information:")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Missing values:\n{df.isnull().sum()}")

# 3. Class Distribution
print("\n3. Class Distribution:")
class_counts = df['Spam/Ham'].value_counts()
print(f"   {class_counts}")

# 4. Text Analysis
print("\n4. Text Analysis:")
df['text_length'] = df['Message'].fillna('').apply(len)
df['word_count'] = df['Message'].fillna('').apply(lambda x: len(str(x).split()))
df['subject_length'] = df['Subject'].fillna('').apply(len)

print(f"   Average text length: {df['text_length'].mean():.0f} chars")
print(f"   Average word count: {df['word_count'].mean():.0f} words")
print(f"   Average subject length: {df['subject_length'].mean():.0f} chars")

# 5. Special Characters
print("\n5. Special Characters Analysis:")
df['exclamation'] = df['Message'].fillna('').apply(lambda x: str(x).count('!'))
df['question'] = df['Message'].fillna('').apply(lambda x: str(x).count('?'))
df['dollar'] = df['Message'].fillna('').apply(lambda x: str(x).count('$'))
df['percent'] = df['Message'].fillna('').apply(lambda x: str(x).count('%'))

special_means = df.groupby('Spam/Ham')[['exclamation', 'question', 'dollar', 'percent']].mean()
print(f"   {special_means}")

# 6. Common Words
print("\n6. Top Words Analysis:")
stop_words = set(stopwords.words('english'))

def get_common_words(texts, n=20):
    words = []
    for text in texts:
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words.extend([w for w in text.split() if w not in stop_words and len(w) > 2])
    return Counter(words).most_common(n)

spam_words = get_common_words(df[df['Spam/Ham']=='spam']['Message'].fillna(''), 10)
ham_words = get_common_words(df[df['Spam/Ham']=='ham']['Message'].fillna(''), 10)

print(f"   Top 5 Spam words: {', '.join([w for w, c in spam_words[:5]])}")
print(f"   Top 5 Ham words: {', '.join([w for w, c in ham_words[:5]])}")

# 7. Create Visualizations
print("\n7. Creating Visualizations...")

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Spam/Ham'].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', 
                                       explode=[0, 0.05], colors=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('')

df['Spam/Ham'].value_counts().plot.bar(ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1].set_title('Class Counts')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
for i, v in enumerate(df['Spam/Ham'].value_counts().values):
    axes[1].text(i, v + 200, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/class_distribution.png")

# Text length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.boxplot(column='text_length', by='Spam/Ham', ax=axes[0])
axes[0].set_title('Text Length by Class')

for label, color in [('ham', '#2ecc71'), ('spam', '#e74c3c')]:
    subset = df[df['Spam/Ham'] == label]['text_length']
    subset.hist(bins=50, alpha=0.5, label=label.capitalize(), color=color, ax=axes[1])
axes[1].set_title('Text Length Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/figures/text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/text_length_distribution.png")

# Common words
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
if spam_words:
    words_spam, counts_spam = zip(*spam_words)
    axes[0].barh(words_spam, counts_spam, color='#e74c3c', edgecolor='black')
    axes[0].set_title('Top 20 Words in SPAM')
    axes[0].invert_yaxis()

if ham_words:
    words_ham, counts_ham = zip(*ham_words)
    axes[1].barh(words_ham, counts_ham, color='#2ecc71', edgecolor='black')
    axes[1].set_title('Top 20 Words in HAM')
    axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('reports/figures/common_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/common_words.png")

# Special characters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
chars = ['exclamation', 'question', 'dollar', 'percent']
titles = ['Exclamation Marks (!)', 'Question Marks (?)', 'Dollar Signs ($)', 'Percent Signs (%)']

for i, (char, title) in enumerate(zip(chars, titles)):
    row, col = i // 2, i % 2
    df.boxplot(column=char, by='Spam/Ham', ax=axes[row, col])
    axes[row, col].set_title(title)

plt.tight_layout()
plt.savefig('reports/figures/special_characters.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/special_characters.png")

# Save dataset stats to file
with open('data/processed/dataset_stats.txt', 'w') as f:
    f.write("ENRON SPAM DATASET STATISTICS\n")
    f.write("="*40 + "\n\n")
    f.write(f"Total emails: {len(df)}\n")
    f.write(f"Ham: {len(df[df['Spam/Ham']=='ham'])}\n")
    f.write(f"Spam: {len(df[df['Spam/Ham']=='spam'])}\n\n")
    f.write("Text Statistics:\n")
    f.write(f"  Avg text length: {df['text_length'].mean():.0f} chars\n")
    f.write(f"  Avg word count: {df['word_count'].mean():.0f} words\n")
    f.write(f"  Avg subject length: {df['subject_length'].mean():.0f} chars\n\n")
    f.write("Special Characters (Avg):\n")
    f.write(f"  Ham - Exclamation: {df[df['Spam/Ham']=='ham']['exclamation'].mean():.2f}\n")
    f.write(f"  Spam - Exclamation: {df[df['Spam/Ham']=='spam']['exclamation'].mean():.2f}\n")
    f.write(f"  Ham - Dollar: {df[df['Spam/Ham']=='ham']['dollar'].mean():.2f}\n")
    f.write(f"  Spam - Dollar: {df[df['Spam/Ham']=='spam']['dollar'].mean():.2f}\n")

print("   ✅ Saved: data/processed/dataset_stats.txt")

# Summary
print("\n" + "="*60)
print("📊 EDA SUMMARY")
print("="*60)
print(f"\nDataset: {len(df)} emails ({len(df[df['Spam/Ham']=='ham'])} ham, {len(df[df['Spam/Ham']=='spam'])} spam)")
print(f"\nKey Findings:")
print(f"   - Spam emails are slightly longer ({df[df['Spam/Ham']=='spam']['text_length'].mean():.0f} chars)")
print(f"   - Ham emails are shorter ({df[df['Spam/Ham']=='ham']['text_length'].mean():.0f} chars)")
print(f"   - Spam has more exclamation marks (!) and dollar signs ($)")
print(f"   - Top spam indicators: {', '.join([w for w, c in spam_words[:3]])}")
print(f"   - Top ham indicators: {', '.join([w for w, c in ham_words[:3]])}")
print("\n" + "="*60)
print("✅ EDA COMPLETE! Ready for preprocessing.")
print("="*60)

# Save processed sample for quick view
df_sample = df[['Subject', 'Message', 'Spam/Ham', 'text_length', 'word_count']].head(100)
df_sample.to_csv('data/processed/eda_sample_100.csv', index=False)
print("\n📁 Sample saved: data/processed/eda_sample_100.csv")
