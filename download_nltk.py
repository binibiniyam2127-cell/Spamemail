import os
import sys
import ssl

# Change to a safe directory to avoid import issues
os.chdir('/tmp')

# Disable SSL verification if needed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import nltk
    print("✅ NLTK imported successfully")
    
    # Download data
    print("📥 Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("✅ All NLTK data downloaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
