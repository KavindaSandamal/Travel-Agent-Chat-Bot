"""
One-shot and Few-shot Learning Implementation for Travel Advisor Chatbot
Demonstrates meta-learning techniques for rapid adaptation to new travel scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning in travel recommendations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Initialize the prototypical network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
        """
        super(PrototypicalNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        """Forward pass through the encoder."""
        return self.encoder(x)
    
    def compute_prototypes(self, support_set: torch.Tensor, support_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute prototypes for each class in the support set.
        
        Args:
            support_set (torch.Tensor): Support set embeddings
            support_labels (torch.Tensor): Support set labels
            
        Returns:
            Dict[int, torch.Tensor]: Prototypes for each class
        """
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_set[mask]
            prototype = torch.mean(class_embeddings, dim=0)
            prototypes[label.item()] = prototype
        
        return prototypes
    
    def compute_distances(self, query_set: torch.Tensor, prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute distances from query set to prototypes.
        
        Args:
            query_set (torch.Tensor): Query set embeddings
            prototypes (Dict[int, torch.Tensor]): Class prototypes
            
        Returns:
            torch.Tensor: Distance matrix
        """
        distances = []
        for query in query_set:
            query_distances = []
            for label, prototype in prototypes.items():
                distance = torch.norm(query - prototype, p=2)
                query_distances.append(distance)
            distances.append(query_distances)
        
        return torch.stack(distances)
    
    def predict(self, query_set: torch.Tensor, support_set: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for query set using prototypical network.
        
        Args:
            query_set (torch.Tensor): Query set
            support_set (torch.Tensor): Support set
            support_labels (torch.Tensor): Support set labels
            
        Returns:
            torch.Tensor: Predicted labels
        """
        # Encode support and query sets
        support_embeddings = self.forward(support_set)
        query_embeddings = self.forward(query_set)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Predict labels (closest prototype)
        predictions = torch.argmin(distances, dim=1)
        
        return predictions

class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for travel recommendation adaptation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        """
        Initialize MAML.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
        """
        super(MAML, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.inner_lr = 0.01
        self.meta_lr = 0.001
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def inner_update(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop update on support set.
        
        Args:
            support_x (torch.Tensor): Support set inputs
            support_y (torch.Tensor): Support set labels
            
        Returns:
            Dict[str, torch.Tensor]: Updated parameters
        """
        # Forward pass
        outputs = self.forward(support_x)
        loss = F.cross_entropy(outputs, support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        
        # Update parameters
        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - self.inner_lr * grad
        
        return updated_params
    
    def apply_updated_params(self, updated_params: Dict[str, torch.Tensor]):
        """Apply updated parameters to the model."""
        for name, param in self.named_parameters():
            param.data = updated_params[name].data
    
    def meta_update(self, query_x: torch.Tensor, query_y: torch.Tensor, 
                   support_x: torch.Tensor, support_y: torch.Tensor) -> float:
        """
        Perform meta update.
        
        Args:
            query_x (torch.Tensor): Query set inputs
            query_y (torch.Tensor): Query set labels
            support_x (torch.Tensor): Support set inputs
            support_y (torch.Tensor): Support set labels
            
        Returns:
            float: Meta loss
        """
        # Inner update
        updated_params = self.inner_update(support_x, support_y)
        
        # Apply updated parameters
        self.apply_updated_params(updated_params)
        
        # Compute meta loss on query set
        query_outputs = self.forward(query_x)
        meta_loss = F.cross_entropy(query_outputs, query_y)
        
        return meta_loss

class TravelFewShotLearner:
    """
    Few-shot learning system for travel recommendations.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the few-shot learner.
        
        Args:
            embedding_model (str): Sentence transformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.prototypical_net = None
        self.maml = None
        
        # Travel categories for few-shot learning
        self.travel_categories = {
            'destinations': ['paris', 'tokyo', 'london', 'new york', 'sydney'],
            'activities': ['sightseeing', 'food tour', 'museum visit', 'shopping', 'adventure'],
            'accommodations': ['hotel', 'hostel', 'apartment', 'resort', 'bnb'],
            'budgets': ['budget', 'mid-range', 'luxury', 'backpacker', 'premium']
        }
    
    def prepare_few_shot_data(self, data: List[Dict], n_way: int = 5, k_shot: int = 5) -> List[Dict]:
        """
        Prepare data for few-shot learning.
        
        Args:
            data (List[Dict]): Travel data
            n_way (int): Number of classes
            k_shot (int): Number of support examples per class
            
        Returns:
            List[Dict]: Few-shot episodes
        """
        episodes = []
        
        # Group data by category
        category_data = {}
        for item in data:
            category = item.get('category', 'general')
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(item)
        
        # Create episodes
        for _ in range(100):  # Generate 100 episodes
            # Sample n_way categories
            selected_categories = np.random.choice(
                list(category_data.keys()), 
                size=min(n_way, len(category_data)), 
                replace=False
            )
            
            episode = {
                'support_set': [],
                'query_set': [],
                'labels': selected_categories.tolist()
            }
            
            for i, category in enumerate(selected_categories):
                category_items = category_data[category]
                
                # Sample k_shot support examples
                support_examples = np.random.choice(
                    category_items, 
                    size=min(k_shot, len(category_items)), 
                    replace=False
                )
                
                # Sample query examples
                remaining_items = [item for item in category_items if item not in support_examples]
                query_examples = np.random.choice(
                    remaining_items, 
                    size=min(3, len(remaining_items)), 
                    replace=False
                )
                
                # Add to episode
                for example in support_examples:
                    episode['support_set'].append({
                        'text': example.get('text', ''),
                        'label': i,
                        'category': category
                    })
                
                for example in query_examples:
                    episode['query_set'].append({
                        'text': example.get('text', ''),
                        'label': i,
                        'category': category
                    })
            
            episodes.append(episode)
        
        return episodes
    
    def train_prototypical_network(self, episodes: List[Dict], epochs: int = 100):
        """
        Train prototypical network on few-shot episodes.
        
        Args:
            episodes (List[Dict]): Few-shot episodes
            epochs (int): Number of training epochs
        """
        print("🎯 Training Prototypical Network for few-shot learning...")
        
        # Initialize network
        embedding_dim = 384  # Sentence transformer dimension
        self.prototypical_net = PrototypicalNetwork(embedding_dim)
        optimizer = optim.Adam(self.prototypical_net.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for episode in episodes:
                # Prepare support and query sets
                support_texts = [item['text'] for item in episode['support_set']]
                support_labels = torch.tensor([item['label'] for item in episode['support_set']])
                
                query_texts = [item['text'] for item in episode['query_set']]
                query_labels = torch.tensor([item['label'] for item in episode['query_set']])
                
                if len(support_texts) == 0 or len(query_texts) == 0:
                    continue
                
                # Encode texts
                support_embeddings = torch.tensor(self.embedding_model.encode(support_texts))
                query_embeddings = torch.tensor(self.embedding_model.encode(query_texts))
                
                # Forward pass
                support_encoded = self.prototypical_net(support_embeddings)
                query_encoded = self.prototypical_net(query_embeddings)
                
                # Compute prototypes
                prototypes = self.prototypical_net.compute_prototypes(support_encoded, support_labels)
                
                # Compute distances
                distances = self.prototypical_net.compute_distances(query_encoded, prototypes)
                
                # Compute loss (negative log probability)
                logits = -distances
                loss = F.cross_entropy(logits, query_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss: {epoch_loss/len(episodes):.4f}")
        
        print("✅ Prototypical Network training completed!")
    
    def train_maml(self, episodes: List[Dict], epochs: int = 100):
        """
        Train MAML on few-shot episodes.
        
        Args:
            episodes (List[Dict]): Few-shot episodes
            epochs (int): Number of training epochs
        """
        print("🎯 Training MAML for few-shot learning...")
        
        # Initialize MAML
        embedding_dim = 384
        output_dim = 10  # Maximum number of classes
        self.maml = MAML(embedding_dim, output_dim=output_dim)
        meta_optimizer = optim.Adam(self.maml.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for episode in episodes:
                # Prepare support and query sets
                support_texts = [item['text'] for item in episode['support_set']]
                support_labels = torch.tensor([item['label'] for item in episode['support_set']])
                
                query_texts = [item['text'] for item in episode['query_set']]
                query_labels = torch.tensor([item['label'] for item in episode['query_set']])
                
                if len(support_texts) == 0 or len(query_texts) == 0:
                    continue
                
                # Encode texts
                support_embeddings = torch.tensor(self.embedding_model.encode(support_texts))
                query_embeddings = torch.tensor(self.embedding_model.encode(query_texts))
                
                # Meta update
                meta_loss = self.maml.meta_update(
                    query_embeddings, query_labels,
                    support_embeddings, support_labels
                )
                
                # Backward pass
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
                
                epoch_loss += meta_loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Meta Loss: {epoch_loss/len(episodes):.4f}")
        
        print("✅ MAML training completed!")
    
    def evaluate_few_shot(self, test_episodes: List[Dict], method: str = 'prototypical') -> Dict[str, float]:
        """
        Evaluate few-shot learning performance.
        
        Args:
            test_episodes (List[Dict]): Test episodes
            method (str): Method to use ('prototypical' or 'maml')
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if method == 'prototypical' and self.prototypical_net is None:
            raise ValueError("Prototypical network not trained")
        elif method == 'maml' and self.maml is None:
            raise ValueError("MAML not trained")
        
        total_correct = 0
        total_samples = 0
        
        for episode in test_episodes:
            # Prepare support and query sets
            support_texts = [item['text'] for item in episode['support_set']]
            support_labels = torch.tensor([item['label'] for item in episode['support_set']])
            
            query_texts = [item['text'] for item in episode['query_set']]
            query_labels = torch.tensor([item['label'] for item in episode['query_set']])
            
            if len(support_texts) == 0 or len(query_texts) == 0:
                continue
            
            # Encode texts
            support_embeddings = torch.tensor(self.embedding_model.encode(support_texts))
            query_embeddings = torch.tensor(self.embedding_model.encode(query_texts))
            
            if method == 'prototypical':
                # Predict using prototypical network
                predictions = self.prototypical_net.predict(
                    query_embeddings, support_embeddings, support_labels
                )
            else:  # MAML
                # Inner update
                updated_params = self.maml.inner_update(support_embeddings, support_labels)
                self.maml.apply_updated_params(updated_params)
                
                # Predict
                predictions = torch.argmax(self.maml(query_embeddings), dim=1)
            
            # Calculate accuracy
            correct = (predictions == query_labels).sum().item()
            total_correct += correct
            total_samples += len(query_labels)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples
        }
    
    def one_shot_learning(self, new_example: str, category: str) -> str:
        """
        Perform one-shot learning for a new travel example.
        
        Args:
            new_example (str): New travel example
            category (str): Category of the example
            
        Returns:
            str: Generated recommendation
        """
        # Encode the new example
        new_embedding = self.embedding_model.encode([new_example])
        
        # Find similar examples in the knowledge base
        # This is a simplified version - in practice, you'd use a more sophisticated approach
        
        if category == 'destination':
            response = f"Based on your interest in {new_example}, I recommend exploring similar destinations with rich cultural heritage and historical significance."
        elif category == 'activity':
            response = f"For {new_example}, I suggest planning your itinerary to include related activities that complement this experience."
        elif category == 'accommodation':
            response = f"Regarding {new_example}, I recommend considering similar accommodation types that offer the same level of comfort and amenities."
        else:
            response = f"Based on {new_example}, I can help you find similar options that match your preferences."
        
        return response

def main():
    """Demonstrate few-shot learning techniques."""
    print("🎯 One-shot and Few-shot Learning Demo")
    print("=" * 50)
    
    # Sample travel data for few-shot learning
    travel_data = [
        {'text': 'I want to visit Paris for its art and culture', 'category': 'destination'},
        {'text': 'Tokyo has amazing food and technology', 'category': 'destination'},
        {'text': 'London is great for museums and history', 'category': 'destination'},
        {'text': 'I love sightseeing in historical cities', 'category': 'activity'},
        {'text': 'Food tours are my favorite travel activity', 'category': 'activity'},
        {'text': 'Museum visits are educational and fun', 'category': 'activity'},
        {'text': 'I prefer staying in luxury hotels', 'category': 'accommodation'},
        {'text': 'Hostels are great for budget travel', 'category': 'accommodation'},
        {'text': 'Apartments offer more space and privacy', 'category': 'accommodation'},
        {'text': 'I travel on a tight budget', 'category': 'budget'},
        {'text': 'Mid-range travel suits my needs', 'category': 'budget'},
        {'text': 'I enjoy luxury travel experiences', 'category': 'budget'},
    ]
    
    # Initialize few-shot learner
    few_shot_learner = TravelFewShotLearner()
    
    # Prepare few-shot episodes
    episodes = few_shot_learner.prepare_few_shot_data(travel_data, n_way=3, k_shot=2)
    print(f"📊 Prepared {len(episodes)} few-shot episodes")
    
    # Split into train and test
    train_episodes = episodes[:80]
    test_episodes = episodes[80:]
    
    # Train Prototypical Network
    few_shot_learner.train_prototypical_network(train_episodes, epochs=50)
    
    # Train MAML
    few_shot_learner.train_maml(train_episodes, epochs=50)
    
    # Evaluate both methods
    print("\n📊 Evaluation Results:")
    print("-" * 30)
    
    # Prototypical Network evaluation
    proto_results = few_shot_learner.evaluate_few_shot(test_episodes, method='prototypical')
    print(f"Prototypical Network Accuracy: {proto_results['accuracy']:.3f}")
    
    # MAML evaluation
    maml_results = few_shot_learner.evaluate_few_shot(test_episodes, method='maml')
    print(f"MAML Accuracy: {maml_results['accuracy']:.3f}")
    
    # One-shot learning demonstration
    print("\n🎯 One-shot Learning Examples:")
    print("-" * 30)
    
    one_shot_examples = [
        ("I want to visit Barcelona for its architecture", "destination"),
        ("I enjoy adventure sports during travel", "activity"),
        ("I prefer boutique hotels for unique experiences", "accommodation")
    ]
    
    for example, category in one_shot_examples:
        response = few_shot_learner.one_shot_learning(example, category)
        print(f"Example: {example}")
        print(f"Category: {category}")
        print(f"Response: {response}")
        print()
    
    print("✅ Few-shot Learning Demo Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Prototypical Networks for few-shot classification")
    print("  - Model-Agnostic Meta-Learning (MAML)")
    print("  - One-shot learning for new travel scenarios")
    print("  - Episode-based training and evaluation")
    print("  - Travel-specific few-shot learning")

if __name__ == "__main__":
    main()
