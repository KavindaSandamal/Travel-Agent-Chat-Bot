"""
Generative AI Implementation for Travel Advisor Chatbot
Demonstrates Autoencoders, RAG, and GANs for travel data generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
try:
    import faiss
except ImportError:
    faiss = None
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None
import json
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TravelAutoencoder(nn.Module):
    """
    Autoencoder for travel data compression and feature learning.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 64):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim (int): Input dimension
            encoding_dim (int): Encoding dimension
        """
        super(TravelAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, encoded):
        """Decode latent representation to output."""
        return self.decoder(encoded)

class TravelGAN(nn.Module):
    """
    Generative Adversarial Network for synthetic travel data generation.
    """
    
    def __init__(self, noise_dim: int = 100, output_dim: int = 50):
        """
        Initialize the GAN.
        
        Args:
            noise_dim (int): Noise vector dimension
            output_dim (int): Output dimension
        """
        super(TravelGAN, self).__init__()
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def generate(self, noise):
        """Generate synthetic data from noise."""
        return self.generator(noise)
    
    def discriminate(self, data):
        """Discriminate between real and fake data."""
        return self.discriminator(data)

class TravelRAGSystem:
    """
    Retrieval-Augmented Generation system for travel recommendations.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model (str): Sentence transformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self.travel_destinations = None
        
        # Initialize ChromaDB (if available)
        if chromadb is not None:
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            try:
                self.collection = self.chroma_client.get_collection("travel_knowledge")
            except:
                self.collection = self.chroma_client.create_collection("travel_knowledge")
        else:
            self.chroma_client = None
            self.collection = None
        
        # Load comprehensive travel destinations dataset
        self._load_travel_destinations()
    
    def _load_travel_destinations(self):
        """Load comprehensive travel destinations dataset efficiently."""
        try:
            # Load the comprehensive travel destinations dataset
            if os.path.exists('data/comprehensive_travel_destinations.csv'):
                self.travel_destinations = pd.read_csv('data/comprehensive_travel_destinations.csv')
                print(f"✅ Loaded {len(self.travel_destinations)} travel destinations from dataset")
                
                # Also load Sri Lanka specific guide if available
                if os.path.exists('data/sri_lanka_travel_guide.csv'):
                    self.sri_lanka_guide = pd.read_csv('data/sri_lanka_travel_guide.csv')
                    print(f"✅ Loaded {len(self.sri_lanka_guide)} Sri Lanka destinations from guide")
                else:
                    self.sri_lanka_guide = None
                
                # Convert to documents for RAG system (sample first 500 for faster loading)
                documents = []
                sample_size = min(500, len(self.travel_destinations))  # Limit to 500 for faster loading
                sample_destinations = self.travel_destinations.head(sample_size)
                
                for idx, row in sample_destinations.iterrows():
                    content = f"Destination: {row['destination']}, Country: {row['country']}, Category: {row['category']}, Attractions: {row['attractions']}, Best Time: {row['best_time']}, Budget: {row['budget']}, Daily Cost: {row['daily_cost']}, Description: {row['description']}"
                    metadata = {
                        'destination': row['destination'],
                        'country': row['country'],
                        'category': row['category'],
                        'rating': row['rating'],
                        'reviews': row['reviews']
                    }
                    documents.append({'content': content, 'metadata': metadata})
                
                # Add to knowledge base
                self.add_documents(documents)
                print(f"✅ Added {len(documents)} documents to knowledge base (sampled for performance)")
                
            else:
                print("⚠️ Comprehensive travel destinations dataset not found")
                self.travel_destinations = None
                
        except Exception as e:
            print(f"⚠️ Error loading travel destinations: {e}")
            self.travel_destinations = None
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'content' and 'metadata'
        """
        print("📚 Adding documents to knowledge base...")
        
        for i, doc in enumerate(documents):
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # Generate embedding
            embedding = self.embedding_model.encode(content)
            
            # Add to ChromaDB (if available)
            if self.collection is not None:
                self.collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[f"doc_{i}"],
                    embeddings=[embedding.tolist()]
                )
            
            # Add to local knowledge base
            self.knowledge_base.append({
                'content': content,
                'metadata': metadata,
                'embedding': embedding
            })
        
        # Build FAISS index
        self._build_faiss_index()
        print(f"✅ Added {len(documents)} documents to knowledge base")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        if not self.knowledge_base or faiss is None:
            return
        
        embeddings = np.array([doc['embedding'] for doc in self.knowledge_base])
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): Search query
            k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: Relevant documents
        """
        if not self.knowledge_base:
            return []
        
        # If we have FAISS index, use it
        if self.index is not None and faiss is not None:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_base):
                    doc = self.knowledge_base[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            return results
        
        # Fallback: simple text-based search
        query_lower = query.lower()
        results = []
        
        for doc in self.knowledge_base:
            content = doc['content'].lower()
            score = 0
            
            # Simple keyword matching
            keywords = query_lower.split()
            for keyword in keywords:
                if keyword in content:
                    score += 1
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy['similarity_score'] = float(score)
                results.append(doc_copy)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:k]
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """
        Generate response using retrieved context and real dataset.
        
        Args:
            query (str): User query
            context (List[Dict]): Retrieved context documents
            
        Returns:
            str: Generated response
        """
        if not context:
            return "I don't have enough information to answer your question."
        
        # Filter context for travel-related content only
        travel_context = []
        for doc in context:
            content = doc.get('content', '')
            # Only include content that seems travel-related and doesn't contain irrelevant keywords
            if (any(keyword in content.lower() for keyword in ['travel', 'destination', 'visit', 'trip', 'tourist', 'attraction', 'hotel', 'restaurant', 'culture', 'city', 'country', 'beach', 'temple', 'museum', 'food', 'cuisine']) and
                not any(irrelevant in content.lower() for irrelevant in ['health', 'education', 'government', 'ministry', 'programme', 'policy', 'research', 'study', 'report', 'analysis'])):
                travel_context.append(content)
        
        # If we have travel destinations dataset, use it to generate responses
        if self.travel_destinations is not None:
            return self._generate_response_from_dataset(query, travel_context)
        
        # Fallback to predefined responses
        return self._get_predefined_travel_response(query)
    
    def _generate_response_from_dataset(self, query: str, context: List[str]) -> str:
        """Generate response using our comprehensive travel destinations dataset."""
        query_lower = query.lower()
        
        # Search for specific destinations in our dataset
        if 'sri lanka' in query_lower:
            sri_lanka_destinations = self.travel_destinations[self.travel_destinations['country'] == 'Sri Lanka']
            if len(sri_lanka_destinations) > 0:
                return self._format_sri_lanka_response(sri_lanka_destinations)
        
        # Search for other specific countries or cities
        for country in ['france', 'japan', 'italy', 'spain', 'germany', 'thailand', 'india', 'china']:
            if country in query_lower:
                country_destinations = self.travel_destinations[
                    self.travel_destinations['country'].str.lower().str.contains(country, na=False)
                ]
                if len(country_destinations) > 0:
                    return self._format_country_response(country_destinations, country.title())
        
        # Search for specific cities
        for idx, row in self.travel_destinations.iterrows():
            destination = row['destination'].lower()
            if destination in query_lower:
                return self._format_destination_response(row)
        
        # If no specific match, provide general recommendations
        return self._format_general_recommendations(query_lower)
    
    def _format_sri_lanka_response(self, sri_lanka_destinations) -> str:
        """Format Sri Lanka destinations response using dedicated guide."""
        response = "🇱🇰 **Best Places to Visit in Sri Lanka**\n\n"
        
        # Use dedicated Sri Lanka guide if available
        if hasattr(self, 'sri_lanka_guide') and self.sri_lanka_guide is not None:
            response += "Based on our dedicated Sri Lanka travel guide, here are the top destinations:\n\n"
            sri_lanka_data = self.sri_lanka_guide
        else:
            response += "Based on our comprehensive travel dataset, here are the top Sri Lanka destinations:\n\n"
            sri_lanka_data = sri_lanka_destinations
        
        # Group by category
        categories = sri_lanka_data['category'].unique()
        for category in categories:
            category_destinations = sri_lanka_data[sri_lanka_data['category'] == category]
            response += f"**{category} Destinations:**\n"
            
            for idx, row in category_destinations.head(5).iterrows():
                response += f"• **{row['destination']}** - {row['attractions']}\n"
                response += f"  Budget: {row['budget']} ({row['daily_cost']}) | Rating: {row['rating']}/5\n"
            
            response += "\n"
        
        response += "**Travel Tips:**\n"
        response += "• Best time: Year-round with dry season (Dec-Mar) preferred\n"
        response += "• Budget range: $25-80/day depending on style\n"
        response += "• Must-try: Local cuisine, tea plantations, cultural sites\n\n"
        response += "What type of experience interests you most?"
        
        return response
    
    def _format_country_response(self, country_destinations, country_name) -> str:
        """Format country destinations response."""
        response = f"🌍 **Top Destinations in {country_name}**\n\n"
        response += f"Based on our travel dataset, here are the best places to visit in {country_name}:\n\n"
        
        # Show top destinations by rating
        top_destinations = country_destinations.nlargest(8, 'rating')
        
        for idx, row in top_destinations.iterrows():
            response += f"**{row['destination']}** ({row['category']})\n"
            response += f"• {row['attractions']}\n"
            response += f"• Budget: {row['budget']} ({row['daily_cost']}) | Rating: {row['rating']}/5 ({row['reviews']} reviews)\n"
            response += f"• Best time: {row['best_time']}\n\n"
        
        return response
    
    def _format_destination_response(self, destination_row) -> str:
        """Format specific destination response."""
        response = f"🏛️ **{destination_row['destination']} Travel Guide**\n\n"
        response += f"**Location:** {destination_row['destination']}, {destination_row['country']}\n"
        response += f"**Category:** {destination_row['category']}\n"
        response += f"**Rating:** {destination_row['rating']}/5 ({destination_row['reviews']} reviews)\n\n"
        
        response += f"**Attractions:**\n{destination_row['attractions']}\n\n"
        
        response += f"**Travel Information:**\n"
        response += f"• Budget: {destination_row['budget']} ({destination_row['daily_cost']})\n"
        response += f"• Best time: {destination_row['best_time']}\n"
        response += f"• Accommodation: {destination_row['accommodation']}\n"
        response += f"• Food: {destination_row['food']}\n\n"
        
        response += f"**Description:**\n{destination_row['description']}\n\n"
        response += "Would you like more specific information about this destination?"
        
        return response
    
    def _format_general_recommendations(self, query_lower) -> str:
        """Format general travel recommendations."""
        # Check if this is a follow-up to a previous recommendation
        if any(keyword in query_lower for keyword in ['natural', 'cultural', 'adventure', 'modern']):
            return self._format_category_specific_recommendations(query_lower)
        
        # Get top-rated destinations
        top_destinations = self.travel_destinations.nlargest(10, 'rating')
        
        response = "🌍 **Travel Recommendations**\n\n"
        response += "Based on our comprehensive travel dataset, here are some top-rated destinations:\n\n"
        
        for idx, row in top_destinations.iterrows():
            response += f"**{row['destination']}, {row['country']}** ({row['category']})\n"
            response += f"• Rating: {row['rating']}/5 | Budget: {row['budget']} ({row['daily_cost']})\n"
            response += f"• {row['attractions'][:100]}...\n\n"
        
        response += "**What type of travel experience are you looking for?**\n"
        response += "• Cultural destinations\n"
        response += "• Natural attractions\n"
        response += "• Adventure activities\n"
        response += "• Modern cities\n\n"
        response += "Please let me know your preferences and I can provide more specific recommendations!"
        
        return response
    
    def _format_category_specific_recommendations(self, query_lower) -> str:
        """Format recommendations based on specific category preference."""
        # Determine category from query
        if 'natural' in query_lower:
            category = 'Natural'
            emoji = "🌿"
        elif 'cultural' in query_lower:
            category = 'Cultural'
            emoji = "🏛️"
        elif 'adventure' in query_lower:
            category = 'Adventure'
            emoji = "🏔️"
        elif 'modern' in query_lower:
            category = 'Modern'
            emoji = "🏙️"
        else:
            category = 'Natural'  # Default
            emoji = "🌿"
        
        # Get destinations in this category
        category_destinations = self.travel_destinations[
            self.travel_destinations['category'] == category
        ].nlargest(8, 'rating')
        
        response = f"{emoji} **{category} Travel Destinations**\n\n"
        response += f"Perfect! Here are the best {category.lower()} destinations from our dataset:\n\n"
        
        if len(category_destinations) > 0:
            for idx, row in category_destinations.iterrows():
                response += f"**{row['destination']}, {row['country']}**\n"
                response += f"• {row['attractions']}\n"
                response += f"• Budget: {row['budget']} ({row['daily_cost']}) | Rating: {row['rating']}/5\n"
                response += f"• Best time: {row['best_time']}\n\n"
        else:
            response += f"No {category.lower()} destinations found in our dataset.\n\n"
        
        response += f"**Would you like more details about any of these {category.lower()} destinations?**\n"
        response += "Just ask me about a specific place and I'll give you detailed information!"
        
        return response
    
    def _get_predefined_travel_response(self, query: str) -> str:
        """Get predefined travel responses when RAG context is not relevant."""
        query_lower = query.lower()
        
        if 'sri lanka' in query_lower or 'best places' in query_lower and 'sri lanka' in query_lower:
            return """🇱🇰 **Best Places to Visit in Sri Lanka**

Sri Lanka offers incredible diversity in a compact island nation. Here are the must-visit destinations:

**🏛️ Cultural & Historical Sites:**
• **Sigiriya** - Ancient rock fortress (UNESCO World Heritage)
• **Anuradhapura** - Ancient capital with sacred Bodhi tree
• **Polonnaruwa** - Medieval capital with impressive ruins
• **Kandy** - Cultural heart with Temple of the Sacred Tooth
• **Galle Fort** - Dutch colonial architecture

**🏔️ Hill Country:**
• **Nuwara Eliya** - "Little England" with tea plantations
• **Ella** - Scenic train rides and hiking trails
• **Kandy to Ella Train** - One of the world's most beautiful train journeys

**🏖️ Coastal Areas:**
• **Mirissa** - Whale watching and beautiful beaches
• **Unawatuna** - Popular beach destination
• **Arugam Bay** - Surfing paradise
• **Trincomalee** - Pristine beaches and diving

**🐘 Wildlife:**
• **Yala National Park** - Leopard and elephant spotting
• **Udawalawe National Park** - Elephant sanctuary
• **Minneriya National Park** - Elephant gathering

**Best Time:** December to March (west/south), May to September (east)
**Duration:** 10-14 days for a comprehensive tour
**Budget:** $40-80/day for comfortable travel

What type of experience interests you most - culture, nature, beaches, or wildlife?"""
        
        elif 'paris' in query_lower:
            return """🇫🇷 **Paris Travel Guide**

Paris, the City of Light, is one of the world's most romantic and culturally rich destinations.

**Top Attractions:**
• **Eiffel Tower** - Iconic iron lattice tower
• **Louvre Museum** - World's largest art museum
• **Notre-Dame Cathedral** - Gothic masterpiece
• **Champs-Élysées** - Famous avenue for shopping
• **Montmartre** - Historic artistic neighborhood

**Best Time to Visit:** April-June and September-November
**Budget:** $150-300/day for comfortable travel
**Must-Try:** Croissants, escargot, French wine, macarons

**Travel Tips:**
• Learn basic French phrases
• Use the Metro for transportation
• Book museum tickets in advance
• Visit local markets for authentic experiences

What specific aspect of Paris interests you most?"""
        
        elif 'tokyo' in query_lower:
            return """🇯🇵 **Tokyo Travel Guide**

Tokyo is a fascinating blend of traditional culture and cutting-edge technology.

**Top Attractions:**
• **Senso-ji Temple** - Tokyo's oldest temple
• **Tokyo Skytree** - Tallest structure in Japan
• **Shibuya Crossing** - World's busiest pedestrian crossing
• **Tsukiji Fish Market** - Fresh seafood and sushi
• **Harajuku** - Youth culture and fashion district

**Best Time to Visit:** March-May (cherry blossoms) or September-November
**Budget:** $120-250/day for comfortable travel
**Must-Try:** Ramen, sushi, tempura, matcha tea

**Travel Tips:**
• Get a JR Pass for train travel
• Learn basic Japanese phrases
• Try capsule hotels for unique experience
• Visit during cherry blossom season

What would you like to know more about Tokyo?"""
        else:
            return """🌍 **Travel Assistance**

I'd be happy to help with your travel query! While I don't have specific information about that topic in my current knowledge base, I can provide general travel advice.

**Popular Travel Destinations I can help with:**
• Sri Lanka - Cultural sites, beaches, wildlife
• Paris - Art, culture, cuisine
• Tokyo - Technology, tradition, food
• London - History, museums, theater
• Rome - Ancient history, art, food

**I can help you with:**
• Destination recommendations
• Best time to visit
• Budget planning
• Must-see attractions
• Travel tips and advice

Could you be more specific about what you're looking for? For example:
- "Best places to visit in [country]"
- "Compare [destination1] and [destination2]"
- "Plan a [duration] trip to [destination]"
- "What's the best time to visit [destination]"

What would you like to know?"""

class TravelDataGenerator:
    """
    Generator for synthetic travel data using GANs.
    """
    
    def __init__(self, data_dim: int = 50):
        """
        Initialize the data generator.
        
        Args:
            data_dim (int): Dimension of travel data features
        """
        self.data_dim = data_dim
        self.gan = TravelGAN(noise_dim=100, output_dim=data_dim)
        self.generator_optimizer = optim.Adam(self.gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
        # Training history
        self.generator_losses = []
        self.discriminator_losses = []
    
    def train_gan(self, real_data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train the GAN on real travel data.
        
        Args:
            real_data (np.ndarray): Real travel data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        print("🎨 Training GAN for travel data generation...")
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_data in dataloader:
                real_batch = batch_data[0]
                batch_size_actual = real_batch.size(0)
                
                # Train Discriminator
                self.discriminator_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size_actual, 1)
                real_output = self.gan.discriminate(real_batch)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size_actual, 100)
                fake_data = self.gan.generate(noise)
                fake_labels = torch.zeros(batch_size_actual, 1)
                fake_output = self.gan.discriminate(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                # Train Generator
                self.generator_optimizer.zero_grad()
                
                noise = torch.randn(batch_size_actual, 100)
                fake_data = self.gan.generate(noise)
                fake_output = self.gan.discriminate(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                
                g_loss.backward()
                self.generator_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            self.generator_losses.append(epoch_g_loss / len(dataloader))
            self.discriminator_losses.append(epoch_d_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: G Loss: {epoch_g_loss/len(dataloader):.4f}, D Loss: {epoch_d_loss/len(dataloader):.4f}")
        
        print("✅ GAN training completed!")
    
    def generate_synthetic_data(self, num_samples: int = 100) -> np.ndarray:
        """
        Generate synthetic travel data.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Generated synthetic data
        """
        with torch.no_grad():
            noise = torch.randn(num_samples, 100)
            synthetic_data = self.gan.generate(noise)
            return synthetic_data.numpy()
    
    def plot_training_history(self):
        """Plot GAN training history."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.generator_losses, label='Generator Loss')
        plt.plot(self.discriminator_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training History')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

class TravelFeatureLearner:
    """
    Feature learning using autoencoders for travel data.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 64):
        """
        Initialize the feature learner.
        
        Args:
            input_dim (int): Input dimension
            encoding_dim (int): Encoding dimension
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = TravelAutoencoder(input_dim, encoding_dim)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.losses = []
    
    def train_autoencoder(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train the autoencoder on travel data.
        
        Args:
            data (np.ndarray): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        print("🧠 Training autoencoder for feature learning...")
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_data in dataloader:
                batch = batch_data[0]
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed, encoded = self.autoencoder(batch)
                loss = self.criterion(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            self.losses.append(epoch_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss: {epoch_loss/len(dataloader):.4f}")
        
        print("✅ Autoencoder training completed!")
    
    def encode_data(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data to latent representation.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Encoded data
        """
        with torch.no_grad():
            encoded = self.autoencoder.encode(torch.FloatTensor(data))
            return encoded.numpy()
    
    def decode_data(self, encoded_data: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to original space.
        
        Args:
            encoded_data (np.ndarray): Encoded data
            
        Returns:
            np.ndarray: Decoded data
        """
        with torch.no_grad():
            decoded = self.autoencoder.decode(torch.FloatTensor(encoded_data))
            return decoded.numpy()
    
    def plot_training_history(self):
        """Plot autoencoder training history."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Autoencoder Training History')
        plt.show()

def main():
    """Demonstrate generative AI techniques."""
    print("🎨 Generative AI Techniques Demo")
    print("=" * 50)
    
    # Sample travel data (simulated)
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic travel data
    real_data = np.random.randn(n_samples, n_features)
    
    print(f"📊 Generated {n_samples} samples with {n_features} features")
    
    # 1. Autoencoder for feature learning
    print("\n🧠 1. Autoencoder Feature Learning:")
    feature_learner = TravelFeatureLearner(n_features, encoding_dim=32)
    feature_learner.train_autoencoder(real_data, epochs=50)
    
    # Encode and decode data
    encoded_data = feature_learner.encode_data(real_data[:10])
    decoded_data = feature_learner.decode_data(encoded_data)
    
    print(f"   Original shape: {real_data[:10].shape}")
    print(f"   Encoded shape: {encoded_data.shape}")
    print(f"   Decoded shape: {decoded_data.shape}")
    print(f"   Reconstruction error: {np.mean((real_data[:10] - decoded_data) ** 2):.4f}")
    
    # 2. GAN for synthetic data generation
    print("\n🎨 2. GAN for Synthetic Data Generation:")
    data_generator = TravelDataGenerator(n_features)
    data_generator.train_gan(real_data, epochs=50)
    
    # Generate synthetic data
    synthetic_data = data_generator.generate_synthetic_data(100)
    print(f"   Generated {synthetic_data.shape[0]} synthetic samples")
    print(f"   Synthetic data shape: {synthetic_data.shape}")
    
    # 3. RAG System
    print("\n📚 3. Retrieval-Augmented Generation (RAG):")
    rag_system = TravelRAGSystem()
    
    # Sample travel documents
    travel_docs = [
        {
            'content': 'Paris is the capital of France, known for its art, fashion, and cuisine. Must-visit attractions include the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.',
            'metadata': {'destination': 'Paris', 'country': 'France', 'type': 'destination_info'}
        },
        {
            'content': 'Tokyo is Japan\'s capital, blending traditional culture with modern technology. Key attractions include Senso-ji Temple, Tokyo Skytree, and Tsukiji Fish Market.',
            'metadata': {'destination': 'Tokyo', 'country': 'Japan', 'type': 'destination_info'}
        },
        {
            'content': 'London is the capital of England, famous for its history, royal heritage, and world-class museums. Must-see places include Buckingham Palace, British Museum, and Tower of London.',
            'metadata': {'destination': 'London', 'country': 'England', 'type': 'destination_info'}
        },
        {
            'content': 'The best time to visit Paris is during spring (April-June) or fall (September-November) when the weather is pleasant and crowds are smaller.',
            'metadata': {'destination': 'Paris', 'type': 'travel_tips'}
        },
        {
            'content': 'Tokyo is best visited in spring for cherry blossoms (March-May) or fall for pleasant weather (September-November).',
            'metadata': {'destination': 'Tokyo', 'type': 'travel_tips'}
        }
    ]
    
    # Add documents to knowledge base
    rag_system.add_documents(travel_docs)
    
    # Test queries
    test_queries = [
        "Tell me about Paris",
        "What's the best time to visit Tokyo?",
        "Compare Paris and London"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        
        # Retrieve relevant documents
        relevant_docs = rag_system.retrieve_relevant_documents(query, k=2)
        
        # Generate response
        response = rag_system.generate_response(query, relevant_docs)
        
        print(f"   Relevant docs: {len(relevant_docs)}")
        for doc in relevant_docs:
            print(f"     - {doc['content'][:80]}... (score: {doc['similarity_score']:.3f})")
        print(f"   Response: {response[:100]}...")
    
    print("\n✅ Generative AI Techniques Demo Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Autoencoder for feature learning and compression")
    print("  - GAN for synthetic data generation")
    print("  - RAG system for context-aware responses")
    print("  - Document retrieval and response generation")
    print("  - Travel-specific knowledge base")

if __name__ == "__main__":
    main()
