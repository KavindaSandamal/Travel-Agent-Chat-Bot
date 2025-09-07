#!/usr/bin/env python3
"""
Test script to verify MLOps structure and organization
"""

import os
import sys
import json
from datetime import datetime

def test_mlops_structure():
    """Test the MLOps directory structure."""
    print("🧪 Testing MLOps Structure")
    print("=" * 40)
    
    # Expected directory structure
    expected_dirs = [
        'mlops',
        'mlops/scripts',
        'mlops/configs',
        'mlops/dashboards',
        'mlops/reports',
        'mlops/tests'
    ]
    
    # Expected files
    expected_files = [
        'mlops/scripts/mlops_integration.py',
        'mlops/scripts/train_chatbot_models.py',
        'mlops/scripts/deploy_chatbot_models.py',
        'mlops/scripts/launch_mlops_system.py',
        'mlops/dashboards/mlops_monitoring_dashboard.py',
        'mlops/configs/mlflow_config.yaml',
        'mlops/configs/deployment_config.yaml',
        'mlops/configs/monitoring_config.yaml',
        'mlops/configs/docker_config.yaml',
        'mlops/configs/requirements_mlops.txt',
        'mlops/README.md',
        'launch_mlops.py',
        'PROJECT_STRUCTURE.md'
    ]
    
    # Test directories
    print("📁 Testing Directories:")
    dir_results = {}
    for dir_path in expected_dirs:
        exists = os.path.exists(dir_path)
        dir_results[dir_path] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {dir_path}")
    
    # Test files
    print("\n📄 Testing Files:")
    file_results = {}
    for file_path in expected_files:
        exists = os.path.exists(file_path)
        file_results[file_path] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
    
    # Summary
    total_dirs = len(expected_dirs)
    total_files = len(expected_files)
    existing_dirs = sum(dir_results.values())
    existing_files = sum(file_results.values())
    
    print(f"\n📊 Structure Test Summary:")
    print(f"   Directories: {existing_dirs}/{total_dirs} ({existing_dirs/total_dirs*100:.1f}%)")
    print(f"   Files: {existing_files}/{total_files} ({existing_files/total_files*100:.1f}%)")
    
    # Overall result
    if existing_dirs == total_dirs and existing_files == total_files:
        print("✅ MLOps structure is complete!")
        return True
    else:
        print("❌ MLOps structure is incomplete!")
        return False

def test_imports():
    """Test importing MLOps components."""
    print("\n🔧 Testing Imports:")
    print("=" * 40)
    
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'src'))
    
    import_tests = [
        ("MLOps Pipeline", "from src.mlops.mlops_pipeline import MLOpsPipeline"),
        ("MLOps Integration", "from mlops.scripts.mlops_integration import TravelChatbotMLOps"),
        ("Model Trainer", "from mlops.scripts.train_chatbot_models import ChatbotModelTrainer"),
        ("Model Deployer", "from mlops.scripts.deploy_chatbot_models import ChatbotModelDeployer"),
        ("Monitoring Dashboard", "from mlops.dashboards.mlops_monitoring_dashboard import MLOpsMonitoringDashboard")
    ]
    
    import_results = {}
    for test_name, import_statement in import_tests:
        try:
            exec(import_statement)
            import_results[test_name] = True
            print(f"   ✅ {test_name}")
        except Exception as e:
            import_results[test_name] = False
            print(f"   ❌ {test_name}: {e}")
    
    # Summary
    successful_imports = sum(import_results.values())
    total_imports = len(import_tests)
    
    print(f"\n📊 Import Test Summary:")
    print(f"   Successful: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")
    
    return successful_imports == total_imports

def test_configurations():
    """Test configuration files."""
    print("\n⚙️ Testing Configurations:")
    print("=" * 40)
    
    config_files = [
        'mlops/configs/mlflow_config.yaml',
        'mlops/configs/deployment_config.yaml',
        'mlops/configs/monitoring_config.yaml',
        'mlops/configs/docker_config.yaml'
    ]
    
    config_results = {}
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        config_results[config_file] = True
                        print(f"   ✅ {config_file}")
                    else:
                        config_results[config_file] = False
                        print(f"   ❌ {config_file}: Empty file")
            else:
                config_results[config_file] = False
                print(f"   ❌ {config_file}: File not found")
        except Exception as e:
            config_results[config_file] = False
            print(f"   ❌ {config_file}: {e}")
    
    # Summary
    successful_configs = sum(config_results.values())
    total_configs = len(config_files)
    
    print(f"\n📊 Configuration Test Summary:")
    print(f"   Valid: {successful_configs}/{total_configs} ({successful_configs/total_configs*100:.1f}%)")
    
    return successful_configs == total_configs

def generate_test_report():
    """Generate a test report."""
    print("\n📋 Generating Test Report:")
    print("=" * 40)
    
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('mlops/reports', exist_ok=True)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'structure_test': test_mlops_structure(),
                'import_test': test_imports(),
                'configuration_test': test_configurations()
            },
            'test_summary': {
                'total_tests': 3,
                'passed_tests': 0,  # Will be updated
                'test_status': 'unknown'
            }
        }
        
        # Update test summary
        passed_tests = sum(report['test_results'].values())
        report['test_summary']['passed_tests'] = passed_tests
        report['test_summary']['test_status'] = 'PASSED' if passed_tests == 3 else 'FAILED'
        
        # Save report
        report_file = f"mlops/reports/mlops_structure_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Test report generated: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"❌ Failed to generate test report: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 MLOps Structure and Organization Test")
    print("=" * 60)
    print("🎯 Testing Travel Advisor Chatbot MLOps System")
    print("=" * 60)
    
    # Run all tests
    structure_ok = test_mlops_structure()
    imports_ok = test_imports()
    configs_ok = test_configurations()
    
    # Generate report
    report_file = generate_test_report()
    
    # Final summary
    print("\n🏆 Test Results Summary")
    print("=" * 40)
    
    all_tests_passed = structure_ok and imports_ok and configs_ok
    
    print(f"Structure Test: {'✅ PASSED' if structure_ok else '❌ FAILED'}")
    print(f"Import Test: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    print(f"Configuration Test: {'✅ PASSED' if configs_ok else '❌ FAILED'}")
    
    if all_tests_passed:
        print("\n🎉 All tests passed! MLOps system is properly organized.")
        print("🚀 You can now run: python launch_mlops.py")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
    
    if report_file:
        print(f"\n📊 Detailed report saved to: {report_file}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit(main())
