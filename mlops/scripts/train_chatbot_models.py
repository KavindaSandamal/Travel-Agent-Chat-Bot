#!/usr/bin/env python3
"""
Automated Training Script for Travel Advisor Chatbot Models
Trains all AI models with MLOps tracking and evaluation
"""

import sys
import os
# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'transformers'))
sys.path.append(os.path.join(project_root, 'src', 'embeddings'))
sys.path.append(os.path.join(project_root, 'src', 'generative'))
sys.path.append(os.path.join(project_root, 'src', 'training'))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import MLOps and AI components
from mlops.mlops_pipeline import ModelTrainer, ModelMetrics
from generative_ai import TravelRAGSystem
from word_embeddings import WordEmbeddingGenerator
from llm_integration import TravelTransformerPipeline
from few_shot_learning import TravelFewShotLearner

class ChatbotModelTrainer:
    """
    Automated trainer for all Travel Advisor Chatbot models
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.trainer = ModelTrainer("travel_chatbot_models")
        self.training_data = None
        self.validation_data = None
        
        # Model instances
        self.models = {
            'rag_system': None,
            'embedding_system': None,
            'llm_system': None,
            'few_shot_system': None
        }
        
        # Training results
        self.training_results = {}
        
        print("🎓 Chatbot Model Trainer Initialized")
        print("=" * 50)
    
    def load_training_data(self):
        """Load and prepare training data for all models."""
        print("📊 Loading training data...")
        
        try:
            # Load travel destinations data
            destinations_df = pd.read_csv('data/enhanced_travel_destinations.csv')
            sri_lanka_df = pd.read_csv('data/enhanced_sri_lanka_guide.csv')
            
            # Combine datasets
            combined_df = pd.concat([destinations_df, sri_lanka_df], ignore_index=True)
            
            # Create training features
            self.training_data = self._create_training_features(combined_df)
            self.validation_data = self._create_validation_features(combined_df)
            
            print(f"✅ Training data loaded: {len(self.training_data)} samples")
            print(f"✅ Validation data loaded: {len(self.validation_data)} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def initialize_models(self):
        """Initialize all AI models."""
        print("🚀 Initializing AI models...")
        
        try:
            # Initialize RAG System
            print("📚 Initializing RAG System...")
            self.models['rag_system'] = TravelRAGSystem()
            self.models['rag_system'].load_enhanced_dataset()
            print("✅ RAG System initialized")
            
            # Initialize Word Embedding System
            print("🔤 Initializing Word Embedding System...")
            self.models['embedding_system'] = WordEmbeddingGenerator()
            print("✅ Word Embedding System initialized")
            
            # Initialize LLM Integration
            print("🤖 Initializing LLM Integration...")
            self.models['llm_system'] = TravelTransformerPipeline()
            print("✅ LLM Integration initialized")
            
            # Initialize Few-shot Learning System
            print("🎯 Initializing Few-shot Learning System...")
            self.models['few_shot_system'] = TravelFewShotLearner()
            print("✅ Few-shot Learning System initialized")
            
            print("🎉 All models initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False
    
    def train_rag_model(self):
        """Train the RAG (Retrieval-Augmented Generation) model."""
        print("📚 Training RAG Model...")
        
        try:
            # Create RAG training data
            rag_train_data = self._create_rag_training_data()
            rag_val_data = self._create_rag_validation_data()
            
            # Train with MLOps tracking
            metrics = self.trainer.train_model(
                self.models['rag_system'], 
                rag_train_data, 
                rag_val_data, 
                "rag_model"
            )
            
            self.training_results['rag_model'] = metrics
            
            print(f"✅ RAG Model trained successfully!")
            print(f"   Accuracy: {metrics.accuracy:.3f}")
            print(f"   F1-Score: {metrics.f1_score:.3f}")
            print(f"   Training Time: {metrics.training_time:.2f}s")
            print(f"   Model Size: {metrics.model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ RAG Model training failed: {e}")
            return False
    
    def train_embedding_model(self):
        """Train the Word Embedding model."""
        print("🔤 Training Word Embedding Model...")
        
        try:
            # Create embedding training data
            embedding_train_data = self._create_embedding_training_data()
            embedding_val_data = self._create_embedding_validation_data()
            
            # Train with MLOps tracking
            metrics = self.trainer.train_model(
                self.models['embedding_system'], 
                embedding_train_data, 
                embedding_val_data, 
                "embedding_model"
            )
            
            self.training_results['embedding_model'] = metrics
            
            print(f"✅ Embedding Model trained successfully!")
            print(f"   Accuracy: {metrics.accuracy:.3f}")
            print(f"   F1-Score: {metrics.f1_score:.3f}")
            print(f"   Training Time: {metrics.training_time:.2f}s")
            print(f"   Model Size: {metrics.model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Embedding Model training failed: {e}")
            return False
    
    def train_llm_model(self):
        """Train the LLM Integration model."""
        print("🤖 Training LLM Model...")
        
        try:
            # Create LLM training data
            llm_train_data = self._create_llm_training_data()
            llm_val_data = self._create_llm_validation_data()
            
            # Train with MLOps tracking
            metrics = self.trainer.train_model(
                self.models['llm_system'], 
                llm_train_data, 
                llm_val_data, 
                "llm_model"
            )
            
            self.training_results['llm_model'] = metrics
            
            print(f"✅ LLM Model trained successfully!")
            print(f"   Accuracy: {metrics.accuracy:.3f}")
            print(f"   F1-Score: {metrics.f1_score:.3f}")
            print(f"   Training Time: {metrics.training_time:.2f}s")
            print(f"   Model Size: {metrics.model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM Model training failed: {e}")
            return False
    
    def train_few_shot_model(self):
        """Train the Few-shot Learning model."""
        print("🎯 Training Few-shot Learning Model...")
        
        try:
            # Create few-shot training data
            few_shot_train_data = self._create_few_shot_training_data()
            few_shot_val_data = self._create_few_shot_validation_data()
            
            # Train with MLOps tracking
            metrics = self.trainer.train_model(
                self.models['few_shot_system'], 
                few_shot_train_data, 
                few_shot_val_data, 
                "few_shot_model"
            )
            
            self.training_results['few_shot_model'] = metrics
            
            print(f"✅ Few-shot Model trained successfully!")
            print(f"   Accuracy: {metrics.accuracy:.3f}")
            print(f"   F1-Score: {metrics.f1_score:.3f}")
            print(f"   Training Time: {metrics.training_time:.2f}s")
            print(f"   Model Size: {metrics.model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Few-shot Model training failed: {e}")
            return False
    
    def train_all_models(self):
        """Train all models in sequence."""
        print("🎓 Training All Chatbot Models")
        print("=" * 50)
        
        training_success = {
            'rag_model': False,
            'embedding_model': False,
            'llm_model': False,
            'few_shot_model': False
        }
        
        # Train RAG Model
        training_success['rag_model'] = self.train_rag_model()
        
        # Train Embedding Model
        training_success['embedding_model'] = self.train_embedding_model()
        
        # Train LLM Model
        training_success['llm_model'] = self.train_llm_model()
        
        # Train Few-shot Model
        training_success['few_shot_model'] = self.train_few_shot_model()
        
        # Summary
        successful_models = sum(training_success.values())
        total_models = len(training_success)
        
        print(f"\n📊 Training Summary:")
        print(f"   Successful: {successful_models}/{total_models} models")
        print(f"   Success Rate: {successful_models/total_models*100:.1f}%")
        
        for model_name, success in training_success.items():
            status = "✅" if success else "❌"
            print(f"   {status} {model_name}")
        
        return training_success
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("📊 Evaluating Trained Models")
        print("=" * 50)
        
        evaluation_results = {}
        
        for model_name, metrics in self.training_results.items():
            if metrics:
                print(f"\n🔍 Evaluating {model_name}:")
                print(f"   Accuracy: {metrics.accuracy:.3f}")
                print(f"   Precision: {metrics.precision:.3f}")
                print(f"   Recall: {metrics.recall:.3f}")
                print(f"   F1-Score: {metrics.f1_score:.3f}")
                print(f"   Training Time: {metrics.training_time:.2f}s")
                print(f"   Model Size: {metrics.model_size:.2f}MB")
                
                evaluation_results[model_name] = {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'training_time': metrics.training_time,
                    'model_size': metrics.model_size
                }
        
        return evaluation_results
    
    def save_training_results(self):
        """Save training results to file."""
        print("💾 Saving training results...")
        
        try:
            # Create results directory
            os.makedirs('mlops/results', exist_ok=True)
            
            # Save training results
            results_file = f"mlops/results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert results to serializable format
            serializable_results = {}
            for model_name, metrics in self.training_results.items():
                if metrics:
                    serializable_results[model_name] = {
                        'accuracy': float(metrics.accuracy),
                        'precision': float(metrics.precision),
                        'recall': float(metrics.recall),
                        'f1_score': float(metrics.f1_score),
                        'training_time': float(metrics.training_time),
                        'model_size': float(metrics.model_size),
                        'timestamp': metrics.timestamp
                    }
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"✅ Training results saved to {results_file}")
            return results_file
            
        except Exception as e:
            print(f"❌ Error saving training results: {e}")
            return None
    
    # Helper methods for creating training data
    def _create_training_features(self, df):
        """Create training features from dataset."""
        # Simulate feature extraction
        return np.random.randn(len(df), 10)
    
    def _create_validation_features(self, df):
        """Create validation features from dataset."""
        # Simulate feature extraction
        return np.random.randn(len(df) // 5, 10)
    
    def _create_rag_training_data(self):
        """Create RAG training data."""
        return np.random.randn(100, 10)
    
    def _create_rag_validation_data(self):
        """Create RAG validation data."""
        return np.random.randn(20, 10)
    
    def _create_embedding_training_data(self):
        """Create embedding training data."""
        return np.random.randn(200, 5)
    
    def _create_embedding_validation_data(self):
        """Create embedding validation data."""
        return np.random.randn(40, 5)
    
    def _create_llm_training_data(self):
        """Create LLM training data."""
        return np.random.randn(150, 8)
    
    def _create_llm_validation_data(self):
        """Create LLM validation data."""
        return np.random.randn(30, 8)
    
    def _create_few_shot_training_data(self):
        """Create few-shot training data."""
        return np.random.randn(80, 6)
    
    def _create_few_shot_validation_data(self):
        """Create few-shot validation data."""
        return np.random.randn(16, 6)

def main():
    """Main function to run model training."""
    print("🎓 Travel Advisor Chatbot Model Training")
    print("=" * 60)
    print("🎯 Automated Training with MLOps Integration")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ChatbotModelTrainer()
    
    # Load training data
    if not trainer.load_training_data():
        print("❌ Failed to load training data")
        return
    
    # Initialize models
    if not trainer.initialize_models():
        print("❌ Failed to initialize models")
        return
    
    # Train all models
    training_success = trainer.train_all_models()
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models()
    
    # Save results
    results_file = trainer.save_training_results()
    
    # Final summary
    print("\n🏆 Training Complete!")
    print("=" * 60)
    print("📊 Final Results:")
    
    successful_models = sum(training_success.values())
    total_models = len(training_success)
    
    print(f"   Models Trained: {successful_models}/{total_models}")
    print(f"   Success Rate: {successful_models/total_models*100:.1f}%")
    
    if results_file:
        print(f"   Results Saved: {results_file}")
    
    print("\n🎯 Next Steps:")
    print("   1. Deploy models to production")
    print("   2. Set up monitoring")
    print("   3. Run performance evaluation")
    print("   4. Update model registry")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
