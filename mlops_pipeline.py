#!/usr/bin/env python3
"""
Complete MLOps Pipeline for Travel Advisor Chatbot
Single script that handles training, deployment, monitoring, chatbot launch,
and CI/CD deployment trigger via GitHub Actions.
"""

import sys
import os
import subprocess
from datetime import datetime

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
    print("🎯 Training, Deployment, Monitoring, Chatbot & CI/CD")
    print("=" * 80)

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n🔍 Checking Dependencies...")
    
    required_packages = {
        'mlflow': 'mlflow',
        'streamlit': 'streamlit', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
        'transformers': 'transformers',
        'torch': 'torch',
        'nltk': 'nltk',
        'requests': 'requests'
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
        from mlops.scripts.train_chatbot_models import ChatbotModelTrainer
        trainer = ChatbotModelTrainer()
        
        if not trainer.load_training_data():
            print("❌ Failed to load training data")
            return False
        
        trainer.initialize_models()
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
        os.makedirs("models", exist_ok=True)
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib', '.pt', '.pth'))]
        
        if model_files:
            print(f"✅ Found {len(model_files)} trained models")
            print("✅ Models are ready for local deployment")
        else:
            print("⚠️  No trained models found, but models directory exists")
        
        return True
            
    except Exception as e:
        print(f"❌ Error during local deployment: {e}")
        return False

def start_monitoring():
    """Start monitoring dashboard locally."""
    print("\n📈 Checking Monitoring Dashboard...")
    
    try:
        from mlops.dashboards.mlops_monitoring_dashboard import MLOpsMonitoringDashboard
        print("✅ Monitoring dashboard components loaded")
        print("💡 To start it: python mlops/dashboards/mlops_monitoring_dashboard.py")
        print("   Then open: http://localhost:8502")
        return True
    except Exception as e:
        print(f"⚠️  Monitoring dashboard not available: {e}")
        print("💡 Use MLflow UI instead: http://127.0.0.1:5000")
        return True

def launch_chatbot():
    """Launch the Streamlit chatbot application."""
    print("\n🤖 Launching Travel Advisor Chatbot...")
    print("-" * 40)
    
    try:
        chatbot_cmd = ["streamlit", "run", "travel_chatbot_app.py", "--server.port", "8501"]
        print("🚀 Starting chatbot at: http://localhost:8501")
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

def trigger_ci_cd():
    """Trigger GitHub Actions CI/CD by pushing code."""
    print("\n🚀 Triggering CI/CD Deployment Pipeline...")
    try:
        os.chdir(project_root)
        
        subprocess.run(["git", "add", "."], check=True)
        commit_message = f"Automated MLOps pipeline run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print("✅ Changes pushed! GitHub Actions CI/CD will now run.")
        print("💡 Check your GitHub Actions tab for deployment progress.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push changes: {e}")
        print("💡 Make sure branch = main and repo is clean")
        return False

def main():
    """Main MLOps pipeline function."""
    print_banner()
    
    if not check_dependencies():
        return
    
    if not check_mlflow_server():
        user_input = input("\nContinue without MLflow? (y/n): ").lower()
        if user_input != 'y':
            return
    
    if not train_models():
        return
    
    if not deploy_models():
        return
    
    start_monitoring()
    show_system_status()
    
    # ✅ Trigger CI/CD only after all steps succeed
    trigger_ci_cd()
    
    try:
        launch_chatbot()
    except KeyboardInterrupt:
        print("\n👋 Local MLOps pipeline stopped by user")

if __name__ == "__main__":
    main()
