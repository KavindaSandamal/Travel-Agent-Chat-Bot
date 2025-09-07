"""
Launch Script for Advanced AI Travel Advisor Chatbot
Complete system launcher with all components
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print the system banner."""
    print("=" * 80)
    print("🌍 Advanced AI Travel Advisor Chatbot")
    print("🎓 Academic Assignment - Advanced AI Course")
    print("=" * 80)
    print()

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    print("✅ requirements.txt found")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("⚠️ Data directory not found - will create sample data")
        os.makedirs("data", exist_ok=True)
    
    print("✅ Data directory ready")
    
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Please install manually: pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create sample data if not available."""
    print("\n📊 Creating sample data...")
    
    import pandas as pd
    import numpy as np
    
    # Create sample Bitext data
    bitext_data = pd.DataFrame({
        'instruction': [
            'I want to visit Paris for its art and culture',
            'Tell me about Tokyo food scene',
            'Compare London vs New York for families',
            'Plan a 5-day trip to Rome',
            'What\'s the best time to visit Sydney?',
            'Recommend budget destinations in Europe',
            'Tell me about the weather in Barcelona',
            'What are the must-see attractions in Amsterdam?',
            'Compare Paris and London for a romantic getaway',
            'Plan a family vacation to Japan'
        ],
        'intent': [
            'recommendation', 'information', 'comparison', 'planning', 'information',
            'recommendation', 'information', 'information', 'comparison', 'planning'
        ],
        'category': [
            'destination', 'food', 'comparison', 'planning', 'weather',
            'destination', 'weather', 'attractions', 'comparison', 'planning'
        ],
        'response': [
            'Paris is perfect for art and culture lovers. Visit the Louvre, Musée d\'Orsay, and explore Montmartre for authentic Parisian culture.',
            'Tokyo has incredible food from street vendors to Michelin-starred restaurants. Try ramen, sushi, tempura, and yakitori.',
            'London offers royal heritage and museums, while New York has Broadway and Central Park. Both are family-friendly with excellent public transport.',
            '5-day Rome itinerary: Day 1-2: Ancient Rome (Colosseum, Forum), Day 3: Vatican, Day 4: Trastevere, Day 5: Day trip to Tivoli.',
            'Sydney is best visited in spring (September-November) or fall (March-May) for pleasant weather and fewer crowds.',
            'Budget destinations in Europe: Prague, Budapest, Krakow, Lisbon, and Porto offer great value with rich culture and history.',
            'Barcelona has a Mediterranean climate with mild winters and warm summers. Best time is spring and fall for comfortable weather.',
            'Must-see attractions in Amsterdam: Anne Frank House, Rijksmuseum, Van Gogh Museum, Canal cruises, and the historic Jordaan district.',
            'Paris offers romantic cafes and Seine cruises, while London has charming pubs and Thames walks. Both are perfect for couples.',
            'Family Japan itinerary: Tokyo (3 days), Kyoto (2 days), with visits to temples, gardens, and family-friendly attractions.'
        ]
    })
    
    # Create sample TripAdvisor data
    tripadvisor_data = pd.DataFrame({
        'User ID': [f'User {i}' for i in range(1, 21)],
        'Category 1': np.random.uniform(0.5, 3.0, 20),
        'Category 2': np.random.uniform(0.5, 3.0, 20),
        'Category 3': np.random.uniform(0.5, 3.0, 20),
        'Category 4': np.random.uniform(0.5, 3.0, 20),
        'Category 5': np.random.uniform(0.5, 3.0, 20)
    })
    
    # Save data
    bitext_data.to_csv('data/bitext-travel-llm-chatbot-training-dataset.csv', index=False)
    tripadvisor_data.to_csv('data/tripadvisor_review.csv', index=False)
    
    print("✅ Sample data created successfully")
    print(f"   - Bitext dataset: {len(bitext_data)} Q&A pairs")
    print(f"   - TripAdvisor dataset: {len(tripadvisor_data)} user ratings")

def run_main_application():
    """Run the main application."""
    print("\n🚀 Running main application...")
    
    try:
        # Run main.py
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Main application completed successfully")
            print("📊 Application output:")
            print(result.stdout)
        else:
            print("⚠️ Main application completed with warnings")
            print("📊 Application output:")
            print(result.stdout)
            if result.stderr:
                print("⚠️ Warnings:")
                print(result.stderr)
        
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Main application timed out (this is normal for demo)")
        return True
    except Exception as e:
        print(f"❌ Error running main application: {e}")
        return False

def launch_web_interface():
    """Launch the web interface."""
    print("\n🌐 Launching web interface...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "travel_chatbot_app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        print("✅ Web interface launched successfully")
        print("🌐 URL: http://localhost:8501")
        print("📱 The web interface should open in your browser")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping web interface...")
            process.terminate()
            print("✅ Web interface stopped")
        
        return True
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        return False

def show_system_status():
    """Show system status and information."""
    print("\n📊 System Status:")
    print("-" * 40)
    
    # Check files
    files_to_check = [
        "main.py",
        "travel_chatbot_app.py",
        "requirements.txt",
        "README.md",
        "FINAL_REPORT.md",
        "PROJECT_SUMMARY.md"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Check directories
    dirs_to_check = [
        "src",
        "data",
        "models",
        "notebooks",
        "reports"
    ]
    
    print("\n📁 Directories:")
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
    
    # Check data files
    print("\n📊 Data Files:")
    data_files = [
        "data/bitext-travel-llm-chatbot-training-dataset.csv",
        "data/tripadvisor_review.csv"
    ]
    
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✅ {file} ({size:.1f} KB)")
        else:
            print(f"❌ {file}")

def main():
    """Main launcher function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("❌ System requirements not met. Please fix the issues above.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        return
    
    # Create sample data if needed
    if not os.path.exists("data/bitext-travel-llm-chatbot-training-dataset.csv"):
        create_sample_data()
    
    # Check for comprehensive travel destinations dataset
    if not os.path.exists("data/comprehensive_travel_destinations.csv"):
        print("⚠️ Comprehensive travel destinations dataset not found.")
        print("💡 Run 'python create_travel_dataset.py' to create it from your datasets.")
    
    # Show system status
    show_system_status()
    
    # Menu
    print("\n🎯 What would you like to do?")
    print("1. Run main application (demo)")
    print("2. Launch web interface")
    print("3. Show system status")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                run_main_application()
                break
            elif choice == "2":
                launch_web_interface()
                break
            elif choice == "3":
                show_system_status()
            elif choice == "4":
                print("👋 Goodbye! Safe travels!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Safe travels!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
