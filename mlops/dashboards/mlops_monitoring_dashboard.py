#!/usr/bin/env python3
"""
MLOps Monitoring Dashboard for Travel Advisor Chatbot
Real-time monitoring and alerting system
"""

import streamlit as st
import sys
import os
# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import MLOps components
from mlops.mlops_pipeline import ModelMonitor, MLOpsPipeline

class MLOpsMonitoringDashboard:
    """
    Real-time monitoring dashboard for MLOps pipeline
    """
    
    def __init__(self):
        """Initialize the monitoring dashboard."""
        self.monitors = {}
        self.metrics_history = []
        self.alerts = []
        
        # Initialize monitors for each model
        self.model_endpoints = {
            'rag_model': 'http://localhost:8000/rag',
            'embedding_model': 'http://localhost:8000/embedding',
            'llm_model': 'http://localhost:8000/llm',
            'few_shot_model': 'http://localhost:8000/fewshot'
        }
        
        print("🔍 MLOps Monitoring Dashboard Initialized")
    
    def setup_page(self):
        """Set up the Streamlit page."""
        st.set_page_config(
            page_title="Travel Chatbot MLOps Monitoring",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        .alert-card {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .success-card {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .error-card {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_header(self):
        """Display the dashboard header."""
        st.markdown("""
        <div class="main-header">
            <h1>🔍 Travel Chatbot MLOps Monitoring Dashboard</h1>
            <p>Real-time monitoring and alerting for AI models</p>
        </div>
        """, unsafe_allow_html=True)
    
    def initialize_monitors(self):
        """Initialize monitors for all models."""
        for model_name, endpoint in self.model_endpoints.items():
            try:
                monitor = ModelMonitor(endpoint)
                self.monitors[model_name] = monitor
                st.success(f"✅ Monitor initialized for {model_name}")
            except Exception as e:
                st.error(f"❌ Failed to initialize monitor for {model_name}: {e}")
    
    def collect_metrics(self):
        """Collect metrics from all monitors."""
        current_metrics = {
            'timestamp': datetime.now(),
            'models': {}
        }
        
        for model_name, monitor in self.monitors.items():
            try:
                metrics = monitor.collect_metrics()
                current_metrics['models'][model_name] = metrics
                
                # Check for alerts
                alerts = monitor.check_alerts(metrics)
                if alerts:
                    for alert in alerts:
                        self.alerts.append({
                            'timestamp': datetime.now(),
                            'model': model_name,
                            'alert': alert,
                            'severity': 'HIGH' if 'CRITICAL' in alert else 'MEDIUM'
                        })
                
            except Exception as e:
                st.error(f"❌ Failed to collect metrics for {model_name}: {e}")
                current_metrics['models'][model_name] = {
                    'error': str(e),
                    'health_status': False
                }
        
        self.metrics_history.append(current_metrics)
        
        # Keep only last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def display_overview_metrics(self):
        """Display overview metrics."""
        st.subheader("📊 System Overview")
        
        if not self.metrics_history:
            st.info("No metrics collected yet. Click 'Start Monitoring' to begin.")
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate overall health
        healthy_models = 0
        total_models = len(latest_metrics['models'])
        
        for model_name, metrics in latest_metrics['models'].items():
            if metrics.get('health_status', False):
                healthy_models += 1
        
        health_percentage = (healthy_models / total_models * 100) if total_models > 0 else 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="System Health",
                value=f"{health_percentage:.1f}%",
                delta=f"{healthy_models}/{total_models} models"
            )
        
        with col2:
            avg_response_time = np.mean([
                metrics.get('response_time', 0) 
                for metrics in latest_metrics['models'].values()
                if 'response_time' in metrics
            ])
            st.metric(
                label="Avg Response Time",
                value=f"{avg_response_time:.2f}s",
                delta="Response time"
            )
        
        with col3:
            total_alerts = len(self.alerts)
            recent_alerts = len([a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)])
            st.metric(
                label="Active Alerts",
                value=total_alerts,
                delta=f"{recent_alerts} in last hour"
            )
        
        with col4:
            total_measurements = len(self.metrics_history)
            st.metric(
                label="Measurements",
                value=total_measurements,
                delta="Total collected"
            )
    
    def display_model_metrics(self):
        """Display detailed metrics for each model."""
        st.subheader("🤖 Model Performance")
        
        if not self.metrics_history:
            return
        
        # Create tabs for each model
        model_names = list(self.model_endpoints.keys())
        tabs = st.tabs(model_names)
        
        for i, model_name in enumerate(model_names):
            with tabs[i]:
                self._display_model_tab(model_name)
    
    def _display_model_tab(self, model_name):
        """Display metrics for a specific model."""
        # Get latest metrics for this model
        if not self.metrics_history:
            st.info("No metrics available")
            return
        
        latest_metrics = self.metrics_history[-1]['models'].get(model_name, {})
        
        if not latest_metrics:
            st.error("No metrics available for this model")
            return
        
        # Display current metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_status = latest_metrics.get('health_status', False)
            status_color = "🟢" if health_status else "🔴"
            st.metric(
                label="Health Status",
                value=f"{status_color} {'Healthy' if health_status else 'Unhealthy'}"
            )
        
        with col2:
            response_time = latest_metrics.get('response_time', 0)
            st.metric(
                label="Response Time",
                value=f"{response_time:.2f}s"
            )
        
        with col3:
            accuracy = latest_metrics.get('model_accuracy', 0)
            st.metric(
                label="Model Accuracy",
                value=f"{accuracy:.3f}"
            )
        
        # Display performance charts
        self._display_model_charts(model_name)
    
    def _display_model_charts(self, model_name):
        """Display performance charts for a model."""
        # Filter metrics for this model
        model_metrics = []
        timestamps = []
        
        for measurement in self.metrics_history:
            if model_name in measurement['models']:
                model_metrics.append(measurement['models'][model_name])
                timestamps.append(measurement['timestamp'])
        
        if not model_metrics:
            st.info("No historical data available")
            return
        
        # Create performance charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'Model Accuracy', 'CPU Usage', 'Memory Usage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Response Time
        response_times = [m.get('response_time', 0) for m in model_metrics]
        fig.add_trace(
            go.Scatter(x=timestamps, y=response_times, name='Response Time', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Model Accuracy
        accuracies = [m.get('model_accuracy', 0) for m in model_metrics]
        fig.add_trace(
            go.Scatter(x=timestamps, y=accuracies, name='Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        
        # CPU Usage
        cpu_usage = [m.get('cpu_usage', 0) for m in model_metrics]
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, name='CPU Usage', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Memory Usage
        memory_usage = [m.get('memory_usage', 0) for m in model_metrics]
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_usage, name='Memory Usage', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text=f"{model_name} Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    def display_alerts(self):
        """Display active alerts."""
        st.subheader("🚨 Active Alerts")
        
        if not self.alerts:
            st.success("✅ No active alerts")
            return
        
        # Group alerts by severity
        critical_alerts = [a for a in self.alerts if a['severity'] == 'HIGH']
        medium_alerts = [a for a in self.alerts if a['severity'] == 'MEDIUM']
        
        # Display critical alerts
        if critical_alerts:
            st.error(f"🚨 {len(critical_alerts)} Critical Alerts")
            for alert in critical_alerts[-5:]:  # Show last 5
                st.markdown(f"""
                <div class="error-card">
                    <strong>{alert['model']}</strong> - {alert['alert']}<br>
                    <small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Display medium alerts
        if medium_alerts:
            st.warning(f"⚠️ {len(medium_alerts)} Medium Alerts")
            for alert in medium_alerts[-5:]:  # Show last 5
                st.markdown(f"""
                <div class="alert-card">
                    <strong>{alert['model']}</strong> - {alert['alert']}<br>
                    <small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def display_sidebar(self):
        """Display sidebar controls."""
        st.sidebar.title("🔧 MLOps Controls")
        
        # Monitoring controls
        st.sidebar.subheader("Monitoring")
        
        if st.sidebar.button("🚀 Start Monitoring"):
            self.initialize_monitors()
            st.sidebar.success("Monitoring started!")
        
        if st.sidebar.button("📊 Collect Metrics"):
            self.collect_metrics()
            st.sidebar.success("Metrics collected!")
        
        if st.sidebar.button("🔄 Auto Refresh"):
            st.rerun()
        
        # Model controls
        st.sidebar.subheader("Model Management")
        
        if st.sidebar.button("🎓 Retrain Models"):
            st.sidebar.info("Model retraining initiated...")
        
        if st.sidebar.button("🚀 Deploy Models"):
            st.sidebar.info("Model deployment initiated...")
        
        # Settings
        st.sidebar.subheader("Settings")
        
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=30
        )
        
        alert_threshold = st.sidebar.slider(
            "Alert Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.8
        )
        
        # System info
        st.sidebar.subheader("System Info")
        st.sidebar.info(f"Models: {len(self.model_endpoints)}")
        st.sidebar.info(f"Measurements: {len(self.metrics_history)}")
        st.sidebar.info(f"Alerts: {len(self.alerts)}")
    
    def run_dashboard(self):
        """Run the monitoring dashboard."""
        self.setup_page()
        self.display_header()
        self.display_sidebar()
        
        # Main content
        self.display_overview_metrics()
        self.display_model_metrics()
        self.display_alerts()
        
        # Auto-refresh
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()

def main():
    """Main function to run the monitoring dashboard."""
    dashboard = MLOpsMonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
