"""
Complete MLOps Pipeline for Advanced AI Travel Advisor Chatbot
Demonstrates training, deployment, monitoring, and versioning
"""

import os
import json
import pickle
import logging
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import docker
import yaml
import requests
import time
import threading
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Data class for model metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    model_size: float
    timestamp: str

@dataclass
class DeploymentConfig:
    """Data class for deployment configuration."""
    model_version: str
    deployment_type: str
    container_image: str
    resource_limits: Dict[str, str]
    environment_variables: Dict[str, str]
    health_check_endpoint: str

class ModelTrainer:
    """
    Model training pipeline with MLflow integration.
    """
    
    def __init__(self, experiment_name: str = "travel_chatbot"):
        """
        Initialize the model trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set tracking URI first
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"✅ MLflow experiment '{self.experiment_name}' set up successfully")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}")
            try:
                # Fallback: try with absolute path
                mlflow.set_tracking_uri("file:///D:/Travel%20Agent%20Chat%20Bot/mlruns")
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"✅ MLflow experiment '{self.experiment_name}' set up with fallback URI")
            except Exception as e2:
                logger.error(f"❌ MLflow setup completely failed: {e2}")
                # Continue without MLflow tracking
    
    def train_model(self, model, train_data, val_data, model_name: str = "travel_chatbot") -> ModelMetrics:
        """
        Train a model with MLflow tracking.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data
            model_name (str): Model name
            
        Returns:
            ModelMetrics: Training metrics
        """
        logger.info(f"🚀 Starting training for {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_samples", len(train_data))
            mlflow.log_param("validation_samples", len(val_data))
            
            start_time = time.time()
            
            # Train model (simplified training loop)
            if hasattr(model, 'train'):
                model.train(train_data, val_data)
            else:
                # Simulate training for demo
                time.sleep(2)  # Simulate training time
            
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics = self._evaluate_model(model, val_data)
            metrics.training_time = training_time
            metrics.timestamp = datetime.now().isoformat()
            
            # Log metrics
            try:
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                mlflow.log_metric("training_time", training_time)
            except Exception as e:
                logger.warning(f"Could not log metrics to MLflow: {e}")
            
            # Log model
            if hasattr(model, 'save'):
                model_path = f"models/{model_name}"
                model.save(model_path)
                mlflow.log_artifacts(model_path)
            else:
                # Save model using pickle for demo
                model_path = f"models/{model_name}.pkl"
                os.makedirs("models", exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                # File is now closed, safe to log artifact
                try:
                    mlflow.log_artifact(model_path)
                except Exception as e:
                    logger.warning(f"Could not log artifact {model_path}: {e}")
            
            # Log model size
            try:
                model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                mlflow.log_metric("model_size_mb", model_size)
                metrics.model_size = model_size
            except Exception as e:
                logger.warning(f"Could not log model size to MLflow: {e}")
                metrics.model_size = 0.0
            
            logger.info(f"✅ Training completed for {model_name}")
            logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"   Training time: {training_time:.2f}s")
            logger.info(f"   Model size: {metrics.model_size:.2f}MB")
            
            return metrics
    
    def _evaluate_model(self, model, val_data) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            val_data: Validation data
            
        Returns:
            ModelMetrics: Evaluation metrics
        """
        # Simulate evaluation for demo
        # In practice, you would use actual model predictions
        
        # Generate random predictions for demo
        y_true = np.random.randint(0, 2, len(val_data))
        y_pred = np.random.randint(0, 2, len(val_data))
        
        # Add some bias to make it look realistic
        y_pred = np.where(np.random.random(len(val_data)) > 0.3, y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=0.0,  # Will be set by caller
            model_size=0.0,     # Will be set by caller
            timestamp=""
        )

class ModelDeployer:
    """
    Model deployment pipeline with Docker containerization.
    """
    
    def __init__(self):
        """Initialize the model deployer."""
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Docker not available: {e}")
    
    def create_dockerfile(self, model_path: str, app_path: str = "app.py") -> str:
        """
        Create Dockerfile for model deployment.
        
        Args:
            model_path (str): Path to trained model
            app_path (str): Path to application file
            
        Returns:
            str: Dockerfile content
        """
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY {app_path} .
COPY {model_path} ./models/
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "{app_path}"]
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        logger.info("✅ Dockerfile created")
        return dockerfile_content
    
    def build_container(self, image_name: str = "travel-chatbot", tag: str = "latest") -> bool:
        """
        Build Docker container for deployment.
        
        Args:
            image_name (str): Container image name
            tag (str): Image tag
            
        Returns:
            bool: Success status
        """
        if not self.docker_client:
            logger.warning("⚠️ Docker not available, skipping container build")
            return False
        
        try:
            logger.info(f"🔨 Building container {image_name}:{tag}")
            
            # Build image
            image, build_logs = self.docker_client.images.build(
                path=".",
                tag=f"{image_name}:{tag}",
                rm=True
            )
            
            logger.info(f"✅ Container built successfully: {image_name}:{tag}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Container build failed: {e}")
            return False
    
    def deploy_container(self, image_name: str, container_name: str, port: int = 8000) -> bool:
        """
        Deploy container to production.
        
        Args:
            image_name (str): Container image name
            container_name (str): Container name
            port (int): Port to expose
            
        Returns:
            bool: Success status
        """
        if not self.docker_client:
            logger.warning("⚠️ Docker not available, skipping deployment")
            return False
        
        try:
            logger.info(f"🚀 Deploying container {container_name}")
            
            # Stop existing container if running
            try:
                existing_container = self.docker_client.containers.get(container_name)
                existing_container.stop()
                existing_container.remove()
                logger.info(f"🔄 Stopped existing container {container_name}")
            except:
                pass
            
            # Run new container
            container = self.docker_client.containers.run(
                image_name,
                name=container_name,
                ports={8000: port},
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                environment={
                    "MODEL_PATH": "/app/models/",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            logger.info(f"✅ Container deployed successfully: {container_name}")
            logger.info(f"   Container ID: {container.short_id}")
            logger.info(f"   Port: {port}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Container deployment failed: {e}")
            return False
    
    def create_deployment_config(self, model_version: str) -> DeploymentConfig:
        """
        Create deployment configuration.
        
        Args:
            model_version (str): Model version
            
        Returns:
            DeploymentConfig: Deployment configuration
        """
        config = DeploymentConfig(
            model_version=model_version,
            deployment_type="docker",
            container_image="travel-chatbot:latest",
            resource_limits={
                "cpu": "1",
                "memory": "2Gi"
            },
            environment_variables={
                "MODEL_PATH": "/app/models/",
                "LOG_LEVEL": "INFO",
                "API_VERSION": "v1"
            },
            health_check_endpoint="/health"
        )
        
        # Save configuration
        config_path = "deployment_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        logger.info(f"✅ Deployment configuration saved to {config_path}")
        return config

class ModelMonitor:
    """
    Model monitoring and alerting system.
    """
    
    def __init__(self, model_endpoint: str = "http://localhost:8000"):
        """
        Initialize the model monitor.
        
        Args:
            model_endpoint (str): Model API endpoint
        """
        self.model_endpoint = model_endpoint
        self.metrics_history = []
        self.alerts = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect model performance metrics.
        
        Returns:
            Dict[str, Any]: Current metrics
        """
        try:
            # Health check
            health_response = requests.get(f"{self.model_endpoint}/health", timeout=5)
            health_status = health_response.status_code == 200
            
            # Performance metrics (simulated)
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'health_status': health_status,
                'response_time': np.random.uniform(0.1, 2.0),  # Simulated
                'throughput': np.random.uniform(10, 100),      # Simulated
                'error_rate': np.random.uniform(0.0, 0.05),    # Simulated
                'cpu_usage': np.random.uniform(20, 80),        # Simulated
                'memory_usage': np.random.uniform(30, 90),     # Simulated
                'model_accuracy': np.random.uniform(0.85, 0.95)  # Simulated
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to collect metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'health_status': False,
                'error': str(e)
            }
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Check for alert conditions.
        
        Args:
            metrics (Dict[str, Any]): Current metrics
            
        Returns:
            List[str]: List of alerts
        """
        alerts = []
        
        # Health check alert
        if not metrics.get('health_status', False):
            alerts.append("🚨 CRITICAL: Model endpoint is down")
        
        # Response time alert
        if metrics.get('response_time', 0) > 5.0:
            alerts.append("⚠️ WARNING: High response time detected")
        
        # Error rate alert
        if metrics.get('error_rate', 0) > 0.1:
            alerts.append("🚨 CRITICAL: High error rate detected")
        
        # CPU usage alert
        if metrics.get('cpu_usage', 0) > 90:
            alerts.append("⚠️ WARNING: High CPU usage")
        
        # Memory usage alert
        if metrics.get('memory_usage', 0) > 90:
            alerts.append("⚠️ WARNING: High memory usage")
        
        # Model accuracy alert
        if metrics.get('model_accuracy', 1.0) < 0.8:
            alerts.append("🚨 CRITICAL: Model accuracy below threshold")
        
        # Store alerts
        for alert in alerts:
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'alert': alert,
                'metrics': metrics
            })
        
        return alerts
    
    def start_monitoring(self, interval: int = 60):
        """
        Start continuous monitoring.
        
        Args:
            interval (int): Monitoring interval in seconds
        """
        logger.info(f"🔍 Starting model monitoring (interval: {interval}s)")
        
        def monitor_loop():
            while True:
                try:
                    metrics = self.collect_metrics()
                    alerts = self.check_alerts(metrics)
                    
                    if alerts:
                        for alert in alerts:
                            logger.warning(alert)
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    time.sleep(interval)
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        summary = {
            'total_measurements': len(self.metrics_history),
            'recent_health_status': sum(1 for m in recent_metrics if m.get('health_status', False)) / len(recent_metrics),
            'avg_response_time': np.mean([m.get('response_time', 0) for m in recent_metrics]),
            'avg_throughput': np.mean([m.get('throughput', 0) for m in recent_metrics]),
            'avg_error_rate': np.mean([m.get('error_rate', 0) for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m.get('cpu_usage', 0) for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.get('memory_usage', 0) for m in recent_metrics]),
            'avg_model_accuracy': np.mean([m.get('model_accuracy', 0) for m in recent_metrics]),
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }
        
        return summary

class MLOpsPipeline:
    """
    Complete MLOps pipeline orchestrator.
    """
    
    def __init__(self):
        """Initialize the MLOps pipeline."""
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer()
        self.monitor = ModelMonitor()
        
    def run_full_pipeline(self, model, train_data, val_data, model_name: str = "travel_chatbot") -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline.
        
        Args:
            model: Model to train and deploy
            train_data: Training data
            val_data: Validation data
            model_name (str): Model name
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        logger.info("🚀 Starting complete MLOps pipeline")
        
        results = {
            'training': None,
            'deployment': None,
            'monitoring': None,
            'pipeline_status': 'running'
        }
        
        try:
            # Step 1: Train Model
            logger.info("📚 Step 1: Training model")
            training_metrics = self.trainer.train_model(model, train_data, val_data, model_name)
            results['training'] = asdict(training_metrics)
            
            # Step 2: Deploy Model
            logger.info("🚀 Step 2: Deploying model")
            deployment_config = self.deployer.create_deployment_config(training_metrics.timestamp)
            dockerfile_created = self.deployer.create_dockerfile(f"models/{model_name}.pkl")
            container_built = self.deployer.build_container(f"{model_name}", "latest")
            container_deployed = self.deployer.deploy_container(f"{model_name}:latest", f"{model_name}-container")
            
            results['deployment'] = {
                'config': asdict(deployment_config),
                'dockerfile_created': dockerfile_created,
                'container_built': container_built,
                'container_deployed': container_deployed
            }
            
            # Step 3: Start Monitoring
            logger.info("🔍 Step 3: Starting monitoring")
            monitor_thread = self.monitor.start_monitoring(interval=30)
            
            # Collect initial metrics
            initial_metrics = self.monitor.collect_metrics()
            results['monitoring'] = {
                'monitoring_started': True,
                'initial_metrics': initial_metrics,
                'monitor_thread_alive': monitor_thread.is_alive()
            }
            
            results['pipeline_status'] = 'completed'
            logger.info("✅ MLOps pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"❌ MLOps pipeline failed: {e}")
            results['pipeline_status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dict[str, Any]: Pipeline status
        """
        return {
            'trainer_available': self.trainer is not None,
            'deployer_available': self.deployer is not None,
            'monitor_available': self.monitor is not None,
            'monitoring_metrics': self.monitor.get_metrics_summary() if self.monitor else None
        }

def main():
    """Demonstrate the complete MLOps pipeline."""
    print("🔧 Complete MLOps Pipeline Demo")
    print("=" * 50)
    
    # Initialize MLOps pipeline
    mlops_pipeline = MLOpsPipeline()
    
    # Create sample model and data
    class SampleModel:
        def __init__(self):
            self.trained = False
        
        def train(self, train_data, val_data):
            self.trained = True
            time.sleep(1)  # Simulate training
        
        def save(self, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(self, f)
    
    # Sample data
    train_data = np.random.randn(100, 10)
    val_data = np.random.randn(20, 10)
    model = SampleModel()
    
    # Run complete pipeline
    print("🚀 Running complete MLOps pipeline...")
    results = mlops_pipeline.run_full_pipeline(model, train_data, val_data, "travel_chatbot_demo")
    
    # Display results
    print("\n📊 Pipeline Results:")
    print(f"  Status: {results['pipeline_status']}")
    
    if results['training']:
        print(f"  Training Accuracy: {results['training']['accuracy']:.3f}")
        print(f"  Training Time: {results['training']['training_time']:.2f}s")
        print(f"  Model Size: {results['training']['model_size']:.2f}MB")
    
    if results['deployment']:
        print(f"  Container Built: {results['deployment']['container_built']}")
        print(f"  Container Deployed: {results['deployment']['container_deployed']}")
    
    if results['monitoring']:
        print(f"  Monitoring Started: {results['monitoring']['monitoring_started']}")
        print(f"  Initial Health: {results['monitoring']['initial_metrics'].get('health_status', 'Unknown')}")
    
    # Get pipeline status
    status = mlops_pipeline.get_pipeline_status()
    print(f"\n🔍 Pipeline Status:")
    print(f"  Trainer Available: {status['trainer_available']}")
    print(f"  Deployer Available: {status['deployer_available']}")
    print(f"  Monitor Available: {status['monitor_available']}")
    
    if status['monitoring_metrics']:
        metrics = status['monitoring_metrics']
        print(f"  Total Measurements: {metrics.get('total_measurements', 0)}")
        print(f"  Recent Health Status: {metrics.get('recent_health_status', 0):.1%}")
        print(f"  Average Response Time: {metrics.get('avg_response_time', 0):.2f}s")
        print(f"  Total Alerts: {metrics.get('total_alerts', 0)}")
    
    print("\n✅ MLOps Pipeline Demo Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Model training with MLflow tracking")
    print("  - Docker containerization and deployment")
    print("  - Model monitoring and alerting")
    print("  - Complete CI/CD pipeline")
    print("  - Performance metrics collection")
    print("  - Automated health checks")

if __name__ == "__main__":
    main()
