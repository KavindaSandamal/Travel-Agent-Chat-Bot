#!/usr/bin/env python3
"""
Run Travel Advisor Chatbot with imported models
For use on new PCs with pre-trained models
"""

import sys
import os
import subprocess
import webbrowser
import time

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

def check_models_available():
    """Check if trained models are available."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib', '.pt', '.pth'))]
    return len(model_files) > 0

def check_data_available():
    """Check if required data files are available."""
    required_files = [
        "data/enhanced_travel_destinations.csv",
        "data/enhanced_sri_lanka_guide.csv"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            return False
    return True

def start_mlflow_server():
    """Start MLflow server if models are available."""
    if os.path.exists("mlflow.db") or os.path.exists("mlruns"):
        print("🔄 Starting MLflow server...")
        mlflow_cmd = "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"
        
        if os.name == 'nt':  # Windows
            cmd = f'start "MLflow Server" cmd /k "{mlflow_cmd}"'
            subprocess.Popen(cmd, shell=True)
            print("📝 MLflow server started in new terminal")
        else:  # Unix-like systems
            cmd = f'xterm -e "{mlflow_cmd}" &'
            subprocess.Popen(cmd, shell=True)
            print("📝 MLflow server started in new terminal")
        
        # Wait for server to start
        time.sleep(3)
        try:
            import requests
            response = requests.get("http://127.0.0.1:5000", timeout=2)
            if response.status_code == 200:
                print("✅ MLflow server is running")
                return True
        except:
            pass
        print("⚠️  MLflow server may still be starting...")
        return True
    return False

def run_chatbot():
    """Run the chatbot application."""
    print("🤖 Starting Travel Advisor Chatbot...")
    print("=" * 50)
    
    # Check if models are available
    if check_models_available():
        print("✅ Found trained models - using pre-trained models")
    else:
        print("⚠️  No trained models found - will use default models")
        print("💡 For best performance, import trained models first")
    
    # Check if data is available
    if check_data_available():
        print("✅ Found travel datasets - full functionality available")
    else:
        print("⚠️  Some data files missing - limited functionality")
        print("💡 Make sure all data files are in the data/ directory")
    
    # Start MLflow server if available
    mlflow_running = start_mlflow_server()
    
    # Start chatbot
    print("\n🚀 Launching chatbot application...")
    print("📱 Chatbot will be available at: http://localhost:8501")
    
    if mlflow_running:
        print("📊 MLflow UI available at: http://127.0.0.1:5000")
    
    # Open browser after a delay
    def open_browser():
        time.sleep(5)
        try:
            webbrowser.open("http://localhost:8501")
            if mlflow_running:
                webbrowser.open("http://127.0.0.1:5000")
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the Streamlit chatbot app
    chatbot_cmd = ["streamlit", "run", "travel_chatbot_app.py", "--server.port", "8501"]
    subprocess.run(chatbot_cmd)

if __name__ == "__main__":
    run_chatbot()
