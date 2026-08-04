#!/bin/bash
echo "🚀 Setting up Spam Detection System"

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
mkdir -p data/raw data/processed models

# Download the Enron dataset (smaller sample for deployment)
echo "📥 Downloading sample dataset..."
curl -L -o data/raw/enron_sample.csv https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv

# Preprocess data
echo "🔧 Preprocessing data..."
python src/preprocess.py

# Train model
echo "🌲 Training model..."
python src/train_final.py

echo "✅ Setup complete!"
