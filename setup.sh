#!/bin/bash
echo "🚀 Setting up Spam Detection System"

# Install pip dependencies
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create directories
mkdir -p data/raw data/processed models

# Download sample dataset (smaller version for deployment)
echo "📥 Downloading sample dataset..."
curl -L -o data/raw/enron_sample.csv https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv

# Preprocess data
echo "🔧 Preprocessing data..."
python src/preprocess.py

# Train model
echo "🌲 Training model..."
python src/train_final.py

echo "✅ Setup complete!"
