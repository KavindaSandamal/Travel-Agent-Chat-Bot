#!/usr/bin/env python3
"""
MLOps Pipeline Launcher
Simple interface to run different MLOps components
"""

import os
import sys
import subprocess

def print_menu():
    """Print the MLOps menu."""
    print("\n" + "=" * 60)
    print("🚀 TRAVEL ADVISOR CHATBOT - LOCAL MLOPS LAUNCHER")
    print("=" * 60)
    print("Choose an option:")
    print("1. 🎯 Complete Local MLOps Pipeline (Training + Deployment + Chatbot)")
    print("2. 🎓 Train Models Only")
    print("3. 🚀 Deploy Models Locally")
    print("4. 📊 Start Monitoring Dashboard")
    print("5. 🤖 Launch Chatbot Only")
    print("6. 📈 Start MLflow Server Only")
    print("7. 🔍 Check System Status")
    print("8. ❌ Exit")
    print("=" * 60)

def run_complete_pipeline():
    """Run the complete MLOps pipeline."""
    print("\n🎯 Starting Complete MLOps Pipeline...")
    subprocess.run([sys.executable, "mlops_pipeline.py"])

def train_models_only():
    """Train models only."""
    print("\n🎓 Training Models...")
    subprocess.run([sys.executable, "mlops/scripts/train_chatbot_models.py"])

def deploy_models_only():
    """Deploy models locally only."""
    print("\n🚀 Deploying Models Locally...")
    # Create models directory for local deployment
    os.makedirs("models", exist_ok=True)
    print("✅ Local deployment setup complete")
    print("💡 Models will be stored in ./models/ directory")

def start_monitoring():
    """Start monitoring dashboard."""
    print("\n📊 Starting Monitoring Dashboard...")
    subprocess.run([sys.executable, "mlops/dashboards/mlops_monitoring_dashboard.py"])

def launch_chatbot():
    """Launch chatbot only."""
    print("\n🤖 Launching Chatbot...")
    subprocess.run(["streamlit", "run", "travel_chatbot_app.py"])

def start_mlflow():
    """Start MLflow server."""
    print("\n📈 Starting MLflow Server...")
    subprocess.run([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "127.0.0.1",
        "--port", "5000"
    ])

def check_system_status():
    """Check system status."""
    print("\n🔍 System Status Check...")
    print("-" * 30)
    
    # Check MLflow
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=2)
        if response.status_code == 200:
            print("✅ MLflow Server: Running (http://127.0.0.1:5000)")
        else:
            print("❌ MLflow Server: Not responding")
    except:
        print("❌ MLflow Server: Not running")
    
    # Check if models exist
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib', '.pt', '.pth'))]
        if model_files:
            print(f"✅ Trained Models: {len(model_files)} models found")
        else:
            print("❌ Trained Models: No models found")
    else:
        print("❌ Trained Models: Models directory not found")
    
    # Check datasets
    if os.path.exists("data/enhanced_travel_destinations.csv"):
        print("✅ Enhanced Dataset: Available")
    else:
        print("❌ Enhanced Dataset: Not found")
    
    print("\n💡 To start the complete system, choose option 1")

def main():
    """Main launcher function."""
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                run_complete_pipeline()
            elif choice == "2":
                train_models_only()
            elif choice == "3":
                deploy_models_only()
            elif choice == "4":
                start_monitoring()
            elif choice == "5":
                launch_chatbot()
            elif choice == "6":
                start_mlflow()
            elif choice == "7":
                check_system_status()
            elif choice == "8":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
