#!/usr/bin/env python3
"""
Import trained models from exported package
For setting up the chatbot on a new PC
"""

import os
import zipfile
import shutil
import sys

def import_models(zip_file_path):
    """Import models from zip file."""
    if not os.path.exists(zip_file_path):
        print(f"❌ File not found: {zip_file_path}")
        return False
    
    print(f"📦 Importing models from: {zip_file_path}")
    
    # Extract zip file
    extract_dir = "imported_models"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    
    print("🗜️ Extracting model package...")
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(extract_dir)
    
    # Move files to correct locations
    print("📁 Setting up model files...")
    
    # Move models directory
    if os.path.exists(os.path.join(extract_dir, "models")):
        if os.path.exists("models"):
            shutil.rmtree("models")
        shutil.move(os.path.join(extract_dir, "models"), "models")
        print("✅ Models directory imported")
    
    # Move MLflow data
    if os.path.exists(os.path.join(extract_dir, "mlruns")):
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
        shutil.move(os.path.join(extract_dir, "mlruns"), "mlruns")
        print("✅ MLflow runs imported")
    
    if os.path.exists(os.path.join(extract_dir, "mlflow.db")):
        if os.path.exists("mlflow.db"):
            os.remove("mlflow.db")
        shutil.move(os.path.join(extract_dir, "mlflow.db"), "mlflow.db")
        print("✅ MLflow database imported")
    
    # Move data files
    if os.path.exists(os.path.join(extract_dir, "data")):
        os.makedirs("data", exist_ok=True)
        for item in os.listdir(os.path.join(extract_dir, "data")):
            src = os.path.join(extract_dir, "data", item)
            dst = os.path.join("data", item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        print("✅ Data files imported")
    
    # Clean up
    shutil.rmtree(extract_dir)
    
    print("✅ Model import complete!")
    print("🚀 You can now run: python run_chatbot_with_models.py")
    return True

def main():
    """Main import function."""
    if len(sys.argv) != 2:
        print("Usage: python import_models.py <model_package.zip>")
        print("Example: python import_models.py travel_chatbot_models_20250908_123456.zip")
        return
    
    zip_file = sys.argv[1]
    import_models(zip_file)

if __name__ == "__main__":
    main()
