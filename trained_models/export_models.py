#!/usr/bin/env python3
"""
Export trained models for transfer to different PCs
Creates a portable model package
"""

import os
import shutil
import zipfile
from datetime import datetime

def export_models():
    """Export all trained models and necessary files."""
    print("📦 Exporting trained models for transfer...")
    
    # Create export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = f"model_export_{timestamp}"
    os.makedirs(export_dir, exist_ok=True)
    
    # Copy models directory
    if os.path.exists("models"):
        print("📁 Copying models directory...")
        shutil.copytree("models", os.path.join(export_dir, "models"))
        print("✅ Models directory copied")
    else:
        print("⚠️  No models directory found")
    
    # Copy MLflow data
    if os.path.exists("mlruns"):
        print("📊 Copying MLflow runs...")
        shutil.copytree("mlruns", os.path.join(export_dir, "mlruns"))
        print("✅ MLflow runs copied")
    
    if os.path.exists("mlflow.db"):
        print("🗄️ Copying MLflow database...")
        shutil.copy2("mlflow.db", os.path.join(export_dir, "mlflow.db"))
        print("✅ MLflow database copied")
    
    # Copy essential data files
    data_files = [
        "data/travel_destinations.csv",
        "data/sri_lanka_guide.csv",
        "data/enhanced_travel_destinations.csv",
        "data/enhanced_sri_lanka_guide.csv"
    ]
    
    os.makedirs(os.path.join(export_dir, "data"), exist_ok=True)
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📄 Copying {file_path}...")
            shutil.copy2(file_path, os.path.join(export_dir, file_path))
    
    # Create requirements file
    print("📋 Creating requirements file...")
    with open(os.path.join(export_dir, "requirements.txt"), "w") as f:
        f.write("""streamlit
pandas
numpy
scikit-learn
sentence-transformers
transformers
torch
nltk
requests
mlflow
""")
    
    # Create setup instructions
    print("📝 Creating setup instructions...")
    with open(os.path.join(export_dir, "SETUP_INSTRUCTIONS.md"), "w") as f:
        f.write("""# Travel Advisor Chatbot - Model Setup

## Quick Setup on New PC

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Chatbot:**
   ```bash
   python run_chatbot_with_models.py
   ```

3. **Access the Chatbot:**
   - Open browser: http://localhost:8501

## What's Included

- ✅ Trained models (RAG, Embeddings, LLM, Few-shot)
- ✅ MLflow experiment data
- ✅ Enhanced travel datasets
- ✅ All necessary dependencies

## Features Available

- 🗺️ Travel destination recommendations
- 🏝️ Sri Lanka specific queries
- 🏔️ Natural and adventure destination filtering
- 📊 Model performance tracking
- 🔍 Advanced search capabilities

## Notes

- Models are pre-trained and ready to use
- No training required on new PC
- All data is included in this package
""")
    
    # Create zip file
    print("🗜️ Creating portable package...")
    zip_filename = f"travel_chatbot_models_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, export_dir)
                zipf.write(file_path, arcname)
    
    # Clean up export directory
    shutil.rmtree(export_dir)
    
    print(f"✅ Model export complete!")
    print(f"📦 Portable package created: {zip_filename}")
    print(f"📏 Package size: {os.path.getsize(zip_filename) / (1024*1024):.1f} MB")
    print("\n🚀 Transfer this file to your new PC and follow SETUP_INSTRUCTIONS.md")

if __name__ == "__main__":
    export_models()
