# 📁 Travel Advisor Chatbot - Project Structure

## 🎯 Overview
Complete project structure for the Advanced AI Travel Advisor Chatbot with organized MLOps integration.

## 📂 Root Directory Structure

```
Travel Agent Chat Bot/
├── 📁 data/                          # Essential dataset files
│   ├── enhanced_travel_destinations.csv    # 6,649+ destinations
│   ├── enhanced_sri_lanka_guide.csv        # 2,435+ Sri Lanka destinations
│   ├── bitext-travel-llm-chatbot-training-dataset.csv
│   ├── Destination Reviews (final).csv
│   ├── hotels.csv
│   ├── Information for Accommodation.xls
│   ├── Reviews.csv
│   └── tripadvisor_review.csv
├── 📁 src/                           # Source code
│   ├── 📁 embeddings/                # Word embedding system
│   ├── 📁 generative/                # Generative AI (RAG)
│   ├── 📁 mlops/                     # MLOps pipeline
│   ├── 📁 nlp/                       # NLP preprocessing
│   ├── 📁 prompts/                   # Prompt engineering
│   ├── 📁 training/                  # Few-shot learning
│   ├── 📁 transformers/              # LLM integration
│   └── 📁 utils/                     # Utilities
├── 📁 mlops/                         # MLOps system (clean)
│   ├── 📁 scripts/                   # MLOps scripts
│   ├── 📁 configs/                   # Configuration files
│   ├── 📁 dashboards/                # Monitoring dashboards
│   ├── 📁 reports/                   # MLOps reports (empty)
│   ├── 📁 results/                   # Training results
│   ├── 📁 tests/                     # MLOps tests
│   └── README.md                     # MLOps documentation
├── 📁 models/                        # Trained models (4 models)
├── 📁 mlruns/                        # MLflow tracking data
├── 📄 main.py                        # Main application
├── 📄 travel_chatbot_app.py          # Streamlit web app
├── 📄 mlops_pipeline.py              # Complete MLOps pipeline
├── 📄 run_mlops.py                   # Interactive MLOps launcher
├── 📄 simple_mlops.py                # Streamlined MLOps pipeline
├── 📄 create_enhanced_travel_dataset.py  # Dataset creation script
├── 📄 requirements.txt               # Dependencies
├── 📄 PROJECT_STRUCTURE.md           # This file
└── 📄 README.md                      # Project documentation
```

## 🔧 MLOps System Structure

### 📁 mlops/scripts/
**Core MLOps execution scripts**

| File | Description |
|------|-------------|
| `train_chatbot_models.py` | Automated model training pipeline |
| `deploy_chatbot_models.py` | Production deployment system |

### 📁 mlops/configs/
**Configuration files for all MLOps components**

| File | Description |
|------|-------------|
| `mlflow_config.yaml` | MLflow experiment tracking configuration |
| `deployment_config.yaml` | Production deployment settings |
| `monitoring_config.yaml` | Real-time monitoring configuration |
| `docker_config.yaml` | Docker containerization settings |
| `requirements_mlops.txt` | MLOps-specific dependencies |

### 📁 mlops/dashboards/
**Monitoring and visualization dashboards**

| File | Description |
|------|-------------|
| `mlops_monitoring_dashboard.py` | Real-time monitoring dashboard |

### 📁 mlops/reports/
**Generated reports and analytics**

| File | Description |
|------|-------------|
| `training_results_*.json` | Model training results |
| `deployment_status_*.json` | Deployment status reports |
| `performance_metrics_*.json` | Performance analytics |

### 📁 mlops/tests/
**MLOps testing and validation**

| File | Description |
|------|-------------|
| `test_mlops_pipeline.py` | MLOps pipeline tests |
| `test_model_training.py` | Model training tests |
| `test_deployment.py` | Deployment tests |

### 📁 mlops/docker/
**Docker containerization and deployment**

| File | Description |
|------|-------------|
| `Dockerfile.base` | Base Docker image |
| `Dockerfile.rag` | RAG model container |
| `Dockerfile.embedding` | Embedding model container |
| `Dockerfile.llm` | LLM model container |
| `Dockerfile.fewshot` | Few-shot model container |
| `Dockerfile.monitoring` | Monitoring dashboard container |
| `Dockerfile.complete` | Complete application container |
| `docker-compose.yml` | Multi-service orchestration |
| `nginx.conf` | Load balancer configuration |
| `rag_app.py` | RAG model API server |
| `embedding_app.py` | Embedding model API server |
| `llm_app.py` | LLM model API server |
| `fewshot_app.py` | Few-shot model API server |
| `build_docker.sh` | Linux/Mac build script |
| `build_docker.bat` | Windows build script |
| `manage_docker.py` | Docker management script |
| `README.md` | Docker documentation |

## 🚀 Quick Start Commands

### 1. Launch MLOps System (Cleaned & Optimized)
```bash
# Complete MLOps pipeline (recommended)
python mlops_pipeline.py

# Interactive launcher with menu options
python run_mlops.py

# Streamlined pipeline (fastest)
python simple_mlops.py
```

### 2. Run Individual Components
```bash
# Train models
python mlops/scripts/train_chatbot_models.py

# Deploy models
python mlops/scripts/deploy_chatbot_models.py

# Launch monitoring
streamlit run mlops/dashboards/mlops_monitoring_dashboard.py
```

### 3. Local Deployment (Docker-Free)
```bash
# Models are automatically deployed locally
# No Docker complexity - direct file access
# Models stored in ./models/ directory
```

### 4. Run Main Chatbot
```bash
# Launch web app
streamlit run travel_chatbot_app.py

# Run CLI version
python main.py
```

## 📊 MLOps Features

### ✅ Implemented Features
- **Model Training**: Automated training with MLflow tracking
- **Local Deployment**: Docker-free local model deployment
- **Real-time Monitoring**: Performance tracking and alerting
- **Model Versioning**: Complete version control and registry
- **MLOps Pipeline**: Streamlined training and deployment
- **Configuration Management**: YAML-based configuration
- **Testing Framework**: Comprehensive testing suite
- **Documentation**: Complete documentation and guides

### 🎯 Advanced AI Course Requirements
- **NLP**: Text preprocessing and word embeddings
- **Transformer Models**: BERT integration and LLM support
- **Generative AI**: RAG system with ChromaDB
- **Model Training**: One-shot and few-shot learning
- **Prompt Engineering**: Dynamic prompt generation
- **MLOps**: Complete production pipeline

## 🔧 Configuration Management

### Environment Variables
```bash
# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=travel_chatbot

# Models
MODEL_PATH=/app/models/
LOG_LEVEL=INFO
API_VERSION=v1

# Monitoring
MONITORING_INTERVAL=30
ALERT_THRESHOLD=0.8
```

### Configuration Files
- `mlops/configs/mlflow_config.yaml` - MLflow settings
- `mlops/configs/deployment_config.yaml` - Deployment settings
- `mlops/configs/monitoring_config.yaml` - Monitoring settings
- `mlops/configs/docker_config.yaml` - Docker settings

## 📈 Performance Metrics

### Expected Performance
- **Response Time**: < 2 seconds
- **Throughput**: > 100 requests/second
- **Accuracy**: > 85% for recommendations
- **Uptime**: > 99.9%

### Monitoring KPIs
- Model accuracy and performance
- System resource utilization
- User satisfaction metrics
- Error rates and response times

## 🧪 Testing Strategy

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and stress testing
- **Model Tests**: Accuracy and performance validation

### Quality Assurance
- Code coverage analysis
- Performance benchmarking
- Model accuracy validation
- System reliability testing

## 🚀 Deployment Architecture

### Production Setup
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   RAG Model     │────│   Database      │
│   (Nginx)       │    │   (Port 8001)   │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ├───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Embedding Model │    │   LLM Model     │    │ Few-shot Model  │
│   (Port 8002)   │    │   (Port 8003)   │    │   (Port 8004)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📚 Documentation

### Available Documentation
- `README.md` - Main project documentation
- `mlops/README.md` - MLOps system documentation
- `PROJECT_STRUCTURE.md` - This file
- Inline code documentation

### Getting Help
1. Check the documentation files
2. Review configuration examples
3. Run the test suite
4. Check error logs

## 🎯 Next Steps

### Immediate Actions
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Launch MLOps System**: `python mlops_pipeline.py`
3. **Access Interfaces**: 
   - MLflow UI: `http://127.0.0.1:5000`
   - Monitoring: `http://localhost:8502`
   - Chatbot: `http://localhost:8501`
4. **Test Components**: Run individual scripts

### Future Enhancements
1. **Cloud Deployment**: Migrate to AWS/Azure/GCP
2. **Advanced Monitoring**: Implement APM tools
3. **AutoML**: Add automated model optimization
4. **Security**: Implement security best practices
5. **Scaling**: Add horizontal scaling capabilities

---

**🎯 Your Travel Advisor Chatbot now has a complete, organized, and production-ready MLOps infrastructure!**
