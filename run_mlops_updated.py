#!/usr/bin/env python3
"""
Updated MLOps Launcher
Uses the updated MLOps pipeline with working monitoring dashboard
"""

import sys
import os
import subprocess

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def print_banner():
    """Print the updated MLOps launcher banner."""
    print("=" * 80)
    print("🚀 TRAVEL ADVISOR CHATBOT - UPDATED MLOps LAUNCHER")
    print("=" * 80)
    print("🎯 Updated Pipeline with Working Monitoring Dashboard")
    print("=" * 80)

def show_menu():
    """Show the launcher menu."""
    print("\n📋 Available Options:")
    print("1. 🚀 Run Complete MLOps Pipeline (Updated)")
    print("2. 📊 Start MLflow UI Only")
    print("3. 📈 Start Working Monitoring Dashboard Only")
    print("4. 🤖 Launch Chatbot Only")
    print("5. 🔧 Check System Status")
    print("6. ❌ Exit")

def run_complete_pipeline():
    """Run the complete updated MLOps pipeline."""
    print("\n🚀 Starting Complete MLOps Pipeline...")
    print("This will:")
    print("  ✅ Start MLflow UI with correct path")
    print("  ✅ Train models with MLOps tracking")
    print("  ✅ Deploy models locally")
    print("  ✅ Start working monitoring dashboard")
    print("  ✅ Open browser links automatically")
    print("  ✅ Launch chatbot application")
    
    try:
        subprocess.run([sys.executable, "mlops_pipeline.py"])
    except KeyboardInterrupt:
        print("\n👋 Pipeline stopped by user")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")

def start_mlflow_only():
    """Start only MLflow UI."""
    print("\n📊 Starting MLflow UI...")
    cmd = "mlflow ui --backend-store-uri file:///D:/Travel%20Agent%20Chat%20Bot/mlruns --default-artifact-root file:///D:/Travel%20Agent%20Chat%20Bot/mlruns --host 127.0.0.1 --port 5000"
    
    if os.name == 'nt':  # Windows
        subprocess.Popen(f'start "MLflow UI" cmd /k "{cmd}"', shell=True)
        print("✅ MLflow UI started in new terminal")
        print("📊 Access at: http://127.0.0.1:5000")
    else:  # Unix-like systems
        subprocess.Popen(f'xterm -e "{cmd}" &', shell=True)
        print("✅ MLflow UI started in new terminal")

def start_monitoring_only():
    """Start only the working monitoring dashboard."""
    print("\n📈 Starting Working Monitoring Dashboard...")
    cmd = "streamlit run working_monitoring_dashboard.py --server.port 8502"
    
    if os.name == 'nt':  # Windows
        subprocess.Popen(f'start "Working Monitoring Dashboard" cmd /k "{cmd}"', shell=True)
        print("✅ Working monitoring dashboard started in new terminal")
        print("📊 Access at: http://localhost:8502")
    else:  # Unix-like systems
        subprocess.Popen(f'xterm -e "{cmd}" &', shell=True)
        print("✅ Working monitoring dashboard started in new terminal")

def launch_chatbot_only():
    """Launch only the chatbot."""
    print("\n🤖 Launching Chatbot...")
    cmd = "streamlit run travel_chatbot_app.py --server.port 8501"
    
    try:
        subprocess.run(cmd.split())
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")

def check_system_status():
    """Check the current system status."""
    print("\n🔧 System Status Check:")
    print("-" * 30)
    
    # Check if MLflow is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=2)
        if response.status_code == 200:
            print("✅ MLflow UI: Running (http://127.0.0.1:5000)")
        else:
            print("❌ MLflow UI: Not responding")
    except:
        print("❌ MLflow UI: Not running")
    
    # Check if monitoring dashboard is running
    try:
        import requests
        response = requests.get("http://localhost:8502", timeout=2)
        if response.status_code == 200:
            print("✅ Working Monitoring Dashboard: Running (http://localhost:8502)")
        else:
            print("❌ Working Monitoring Dashboard: Not responding")
    except:
        print("❌ Working Monitoring Dashboard: Not running")
    
    # Check if chatbot is running
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=2)
        if response.status_code == 200:
            print("✅ Chatbot: Running (http://localhost:8501)")
        else:
            print("❌ Chatbot: Not responding")
    except:
        print("❌ Chatbot: Not running")
    
    # Check model files
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith(('.pkl', '.joblib', '.pt', '.pth'))]
        print(f"✅ Model Files: {len(model_files)} found")
    else:
        print("❌ Model Files: No models directory")
    
    # Check experiments
    if os.path.exists("mlruns"):
        experiments = [d for d in os.listdir("mlruns") if d.isdigit()]
        print(f"✅ Experiments: {len(experiments)} found")
    else:
        print("❌ Experiments: No mlruns directory")

def main():
    """Main launcher function."""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Select an option (1-6): ").strip()
            
            if choice == "1":
                run_complete_pipeline()
            elif choice == "2":
                start_mlflow_only()
            elif choice == "3":
                start_monitoring_only()
            elif choice == "4":
                launch_chatbot_only()
            elif choice == "5":
                check_system_status()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
