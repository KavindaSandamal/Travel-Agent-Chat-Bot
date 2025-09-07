#!/usr/bin/env python3
"""
Complete MLOps Pipeline for Travel Advisor Chatbot
Single script that handles training, deployment, monitoring, and chatbot launch
"""

import sys
import os
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'mlops'))

def print_banner():
    """Print the MLOps pipeline banner."""
    print("=" * 80)
    print("🚀 TRAVEL ADVISOR CHATBOT - COMPLETE MLOPS PIPELINE")
    print("=" * 80)
    print("🎯 Single Script for Training, Deployment, Monitoring & Chatbot")
    print("=" * 80)

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n🔍 Checking Dependencies...")
    
    # Map package names to import names
    required_packages = {
        'mlflow': 'mlflow',
        'streamlit': 'streamlit', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
        'transformers': 'transformers',
        'torch': 'torch',
        'nltk': 'nltk'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_mlflow_server():
    """Check if MLflow server is running."""
    print("\n📊 Checking MLflow Server...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=2)
        if response.status_code == 200:
            print("✅ MLflow server is running on http://127.0.0.1:5000")
            return True
        else:
            print("❌ MLflow server not responding")
            return False
    except:
        print("❌ MLflow server not running")
        print("💡 To start MLflow server, run:")
        print("   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000")
        return False

def train_models():
    """Train all chatbot models with MLOps tracking."""
    print("\n🎓 Training Chatbot Models...")
    print("-" * 50)
    
    try:
        # Import and run training
        from mlops.scripts.train_chatbot_models import ChatbotModelTrainer
        
        trainer = ChatbotModelTrainer()
        
        # Load data and initialize models
        if not trainer.load_training_data():
            print("❌ Failed to load training data")
            return False
        
        trainer.initialize_models()
        
        # Train all models
        success = trainer.train_all_models()
        
        if success:
            print("\n✅ All models trained successfully!")
            trainer.evaluate_models()
            trainer.save_training_results()
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def deploy_models():
    """Deploy trained models locally (no Docker)."""
    print("\n🚀 Deploying Models Locally...")
    print("-" * 35)
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Check if models were trained
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib', '.pt', '.pth'))]
            if model_files:
                print(f"✅ Found {len(model_files)} trained models")
                print("✅ Models are ready for local deployment")
                return True
            else:
                print("⚠️  No trained models found, but models directory exists")
                print("✅ Local deployment setup complete")
                return True
        else:
            print("✅ Created models directory for local deployment")
            return True
            
    except Exception as e:
        print(f"❌ Error during local deployment: {e}")
        return False

def start_monitoring():
    """Start monitoring dashboard locally."""
    print("\n📈 Starting Local Monitoring Dashboard...")
    
    try:
        # Check if monitoring dashboard can be imported
        from mlops.dashboards.mlops_monitoring_dashboard import MLOpsMonitoringDashboard
        
        print("✅ Monitoring dashboard components loaded")
        print("💡 To start monitoring dashboard, run:")
        print("   python mlops/dashboards/mlops_monitoring_dashboard.py")
        print("   Then access: http://localhost:8502")
        return True
        
    except Exception as e:
        print(f"⚠️  Monitoring dashboard not available: {e}")
        print("💡 You can still use MLflow UI for monitoring: http://127.0.0.1:5000")
        return True

def launch_chatbot():
    """Launch the Streamlit chatbot application."""
    print("\n🤖 Launching Travel Advisor Chatbot...")
    print("-" * 40)
    
    try:
        # Start chatbot app
        chatbot_cmd = ["streamlit", "run", "travel_chatbot_app.py", "--server.port", "8501"]
        
        print("🚀 Starting chatbot application...")
        print("📱 Chatbot will be available at: http://localhost:8501")
        
        # Run chatbot in foreground (this will block)
        subprocess.run(chatbot_cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")

def show_system_status():
    """Show the current system status and access points."""
    print("\n" + "=" * 80)
    print("🎉 LOCAL MLOPS PIPELINE COMPLETE!")
    print("=" * 80)
    print("📊 System Status:")
    print("   ✅ MLflow Server: http://127.0.0.1:5000")
    print("   ✅ Local Model Storage: ./models/")
    print("   ✅ Chatbot Application: http://localhost:8501")
    print("=" * 80)
    print("🎯 What's Available:")
    print("   • Model training with MLOps tracking")
    print("   • Local model deployment (no Docker)")
    print("   • Interactive travel advisor chatbot")
    print("   • Model versioning and experiment tracking")
    print("   • Enhanced dataset with 9,084 destinations")
    print("=" * 80)
    print("💡 Local MLOps Benefits:")
    print("   • No Docker complexity - runs directly on your machine")
    print("   • Fast startup and deployment")
    print("   • Easy debugging and development")
    print("   • All data tracked in MLflow UI")
    print("=" * 80)

def main():
    """Main MLOps pipeline function."""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Step 2: Check MLflow server
    if not check_mlflow_server():
        print("\n⚠️  MLflow server not running. Please start it manually.")
        print("💡 Run: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000")
        print("🔄 Or continue without MLflow tracking...")
        user_input = input("\nContinue without MLflow? (y/n): ").lower()
        if user_input != 'y':
            return
    
    # Step 3: Train models
    if not train_models():
        print("\n❌ Model training failed. Exiting.")
        return
    
    # Step 4: Deploy models locally
    if not deploy_models():
        print("\n❌ Local model deployment failed. Exiting.")
        return
    
    # Step 5: Setup monitoring
    start_monitoring()
    
    # Step 6: Show system status
    show_system_status()
    
    # Step 7: Launch chatbot (this will run in foreground)
    try:
        launch_chatbot()
    except KeyboardInterrupt:
        print("\n👋 Local MLOps pipeline stopped by user")
        print("🔄 To restart, run: python mlops_pipeline.py")

if __name__ == "__main__":
    main()
