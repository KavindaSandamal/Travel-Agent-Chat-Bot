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
        
        # Note: Dataset loading is now controlled by main.py to use enhanced dataset
        # self._load_travel_destinations()  # Commented out to allow main.py to control dataset loading
    
    def load_enhanced_dataset(self):
        """Load the enhanced travel destinations dataset."""
        self._load_travel_destinations()
    
    def _load_travel_destinations(self):
        """Load comprehensive travel destinations dataset efficiently."""
        try:
            # Load the comprehensive travel destinations dataset
            # Try to load enhanced dataset first, fallback to comprehensive dataset
            if os.path.exists('data/enhanced_travel_destinations.csv'):
                self.travel_destinations = pd.read_csv('data/enhanced_travel_destinations.csv')
                print(f"✅ Loaded {len(self.travel_destinations)} enhanced travel destinations from dataset")
                
                # Also load enhanced Sri Lanka specific guide if available
                if os.path.exists('data/enhanced_sri_lanka_guide.csv'):
                    self.sri_lanka_guide = pd.read_csv('data/enhanced_sri_lanka_guide.csv')
                    print(f"✅ Loaded {len(self.sri_lanka_guide)} enhanced Sri Lanka destinations from guide")
                else:
                    self.sri_lanka_guide = None
            elif os.path.exists('data/comprehensive_travel_destinations.csv'):
                self.travel_destinations = pd.read_csv('data/comprehensive_travel_destinations.csv')
                print(f"✅ Loaded {len(self.travel_destinations)} travel destinations from dataset")
                
                # Also load Sri Lanka specific guide if available
                if os.path.exists('data/sri_lanka_travel_guide.csv'):
                    self.sri_lanka_guide = pd.read_csv('data/sri_lanka_travel_guide.csv')
                    print(f"✅ Loaded {len(self.sri_lanka_guide)} Sri Lanka destinations from guide")
                else:
                    self.sri_lanka_guide = None
            else:
                print("⚠️ No travel destinations dataset found")
                self.travel_destinations = None
                self.sri_lanka_guide = None
            
            # Convert to documents for RAG system (sample first 500 for faster loading)
            if self.travel_destinations is not None:
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
    
    def generate_response(self, query: str, context: List[Dict], conversation_context: List[str] = None) -> str:
        """
        Generate response using retrieved context and real dataset with conversation history.
        
        Args:
            query (str): User query
            context (List[Dict]): Retrieved context documents
            conversation_context (List[str]): Previous conversation messages for context
            
        Returns:
            str: Generated response
        """
        # Filter context for travel-related content only
        travel_context = []
        if context:
            for doc in context:
                # Handle both string and dictionary context
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                else:
                    content = str(doc)
                
                # Only include content that seems travel-related and doesn't contain irrelevant keywords
                if (any(keyword in content.lower() for keyword in ['travel', 'destination', 'visit', 'trip', 'tourist', 'attraction', 'hotel', 'restaurant', 'culture', 'city', 'country', 'beach', 'temple', 'museum', 'food', 'cuisine']) and
                    not any(irrelevant in content.lower() for irrelevant in ['health', 'education', 'government', 'ministry', 'programme', 'policy', 'research', 'study', 'report', 'analysis'])):
                    travel_context.append(content)
        
        # If we have travel destinations dataset, use it to generate responses
        if self.travel_destinations is not None:
            if conversation_context:
                return self._generate_response_from_dataset(query, travel_context, conversation_context)
            else:
                return self._generate_response_from_dataset(query, travel_context)
        
        # Fallback to predefined responses
        return self._get_predefined_travel_response(query)
    
    def _generate_response_from_dataset(self, query: str, context: List[str], conversation_context: List[str] = None) -> str:
        """Generate response using our comprehensive travel destinations dataset."""
        query_lower = query.lower()
        
        # Combine RAG context with conversation context
        full_context = context.copy()
        if conversation_context:
            full_context.extend(conversation_context)
        
        # Extract conversation context and filters
        conversation_filters = self._extract_conversation_filters(query_lower, full_context)
        
        # Search for specific destinations in our dataset
        # Only trigger Sri Lanka logic if explicitly mentioned, not for city names
        if 'sri lanka' in query_lower and not any(city in query_lower for city in ['kandy', 'colombo', 'galle', 'anuradhapura', 'polonnaruwa', 'sigiriya', 'ella', 'nuwara eliya', 'trincomalee', 'batticaloa', 'jaffna', 'negombo', 'bentota', 'hikkaduwa', 'unawatuna', 'mirissa', 'tangalle', 'arugam bay']):
            sri_lanka_destinations = self.travel_destinations[self.travel_destinations['country'] == 'Sri Lanka']
            if len(sri_lanka_destinations) > 0:
                # Check if this is a specific category request (like beaches or mountains)
                if any(keyword in query_lower for keyword in ['beach', 'beaches', 'coastal', 'seaside', 'ocean', 'sea', 'surfing', 'diving', 'snorkeling']):
                    # Apply beach filtering to Sri Lanka destinations
                    conversation_filters['region'] = 'Sri Lanka'
                    conversation_filters['category'] = 'Beach'
                    filtered_destinations = self._apply_comprehensive_filters(conversation_filters)
                    if len(filtered_destinations) > 0:
                        return self._format_category_specific_recommendations(query_lower, conversation_filters)
                elif any(keyword in query_lower for keyword in ['mountain', 'mountains', 'hill', 'hills', 'peak', 'peaks', 'summit', 'summits', 'range', 'ranges', 'ridge', 'ridges', 'knuckles', 'adam', 'pidurutalagala', 'kirigalpotta', 'totapolakanda', 'hakgala', 'horton', 'world\'s end', 'ella rock', 'little adam\'s peak']):
                    # Apply mountain filtering to Sri Lanka destinations
                    conversation_filters['region'] = 'Sri Lanka'
                    conversation_filters['category'] = 'Natural'
                    conversation_filters['interests'] = ['Mountains']  # Set interests to Mountains
                    filtered_destinations = self._apply_comprehensive_filters(conversation_filters)
                    if len(filtered_destinations) > 0:
                        return self._format_category_specific_recommendations(query_lower, conversation_filters)
                # Otherwise return general Sri Lanka response
                return self._format_sri_lanka_response(sri_lanka_destinations)
        
        # Search for other specific countries or cities
        for country in ['france', 'japan', 'italy', 'spain', 'germany', 'thailand', 'india', 'china', 'egypt', 'kenya', 'morocco', 'nigeria', 'south africa', 'australia', 'canada', 'brazil', 'argentina', 'mexico', 'russia', 'turkey', 'saudi arabia', 'uae', 'israel', 'jordan', 'lebanon', 'iran', 'iraq', 'syria', 'afghanistan', 'pakistan', 'bangladesh', 'sri lanka', 'nepal', 'bhutan', 'myanmar', 'laos', 'cambodia', 'vietnam', 'malaysia', 'indonesia', 'philippines', 'brunei', 'mongolia', 'kazakhstan', 'kyrgyzstan', 'tajikistan', 'turkmenistan', 'uzbekistan']:
            if country in query_lower:
                country_destinations = self.travel_destinations[
                    self.travel_destinations['country'].str.lower().str.contains(country, na=False)
                ]
                if len(country_destinations) > 0:
                    return self._format_country_response(country_destinations, country.title())
        
        # Handle region-based queries (like "places to visit in africa")
        # Only process region-based queries if no specific city is mentioned
        if conversation_filters.get('region') and not any(city in query_lower for city in ['kandy', 'colombo', 'galle', 'anuradhapura', 'polonnaruwa', 'sigiriya', 'ella', 'nuwara eliya', 'los angeles', 'new york', 'london', 'paris', 'tokyo', 'rome', 'madrid', 'berlin', 'bangkok', 'mumbai', 'beijing', 'sydney', 'toronto', 'dubai', 'singapore', 'seoul', 'amsterdam', 'vienna', 'prague', 'budapest', 'istanbul', 'cairo', 'cape town', 'nairobi', 'lagos', 'rio de janeiro', 'buenos aires', 'mexico city']):
            region = conversation_filters['region']
            if region in ['africa', 'asia', 'europe', 'america', 'oceania']:
                # Get destinations from that region
                region_destinations = self._filter_destinations_by_region(region)
                if len(region_destinations) > 0:
                    return self._format_country_response(region_destinations, region.title())
        
        # Search for specific cities - prioritize city matches even if region is detected
        city_matches = []
        for idx, row in self.travel_destinations.iterrows():
            destination = row['destination'].lower()
            # Match if the destination name appears in the query (more flexible matching)
            if destination in query_lower or query_lower in destination:
                city_matches.append(row)
        
        # If we found specific city matches, handle them
        if city_matches:
            # If it's a general query like "places to visit in kandy", show multiple destinations
            if any(keyword in query_lower for keyword in ['places to visit', 'destinations in', 'things to see', 'attractions in']):
                # Filter to destinations in the same city/country as the first match
                first_match = city_matches[0]
                city_name = first_match['destination']
                country_name = first_match['country']
                
                # Get all destinations in the same city/country
                city_destinations = self.travel_destinations[
                    (self.travel_destinations['destination'].str.lower().str.contains(city_name.lower(), na=False)) |
                    (self.travel_destinations['country'] == country_name)
                ]
                
                if len(city_destinations) > 0:
                    return self._format_country_response(city_destinations, city_name)
            else:
                # For specific destination queries, return the first match
                return self._format_destination_response(city_matches[0])
        
        # If no city matches found but query mentions a specific city, provide helpful message
        if any(keyword in query_lower for keyword in ['places to visit', 'destinations in', 'things to see', 'attractions in']):
            # Check if query mentions a city that's not in our dataset
            potential_cities = ['los angeles', 'new york', 'london', 'paris', 'tokyo', 'sydney', 'toronto', 'mumbai', 'dubai', 'singapore', 'bangkok', 'seoul', 'berlin', 'rome', 'madrid', 'amsterdam', 'vienna', 'prague', 'budapest', 'istanbul', 'cairo', 'cape town', 'nairobi', 'lagos', 'rio de janeiro', 'buenos aires', 'mexico city', 'lima', 'santiago', 'bogota', 'caracas', 'havana', 'kingston', 'nassau', 'san juan', 'ottawa', 'vancouver', 'montreal', 'calgary', 'edmonton', 'winnipeg', 'halifax', 'victoria', 'quebec city', 'hamilton', 'london', 'manchester', 'birmingham', 'glasgow', 'edinburgh', 'belfast', 'cardiff', 'bristol', 'liverpool', 'leeds', 'sheffield', 'newcastle', 'nottingham', 'leicester', 'coventry', 'bradford', 'hull', 'plymouth', 'stoke', 'wolverhampton', 'derby', 'swansea', 'southampton', 'salford', 'aberdeen', 'westminster', 'portsmouth', 'york', 'peterborough', 'dundee', 'lancaster', 'oxford', 'newport', 'preston', 'st albans', 'norwich', 'chester', 'cambridge', 'exeter', 'gloucester', 'bath', 'ipswich', 'brighton', 'blackpool', 'middlesbrough', 'bolton', 'stockport', 'oldham', 'rotherham', 'swindon', 'grimsby', 'huddersfield', 'poole', 'southport', 'birkenhead', 'worcester', 'hartlepool', 'halifax', 'wigan', 'rhondda', 'southend', 'maidstone', 'eastbourne', 'weston', 'tamworth', 'watford', 'macclesfield', 'rochdale', 'solihull', 'northampton', 'nuneaton', 'darlington', 'barry', 'stevenage', 'hartlepool', 'hemel hempstead', 'saint helens', 'burnley', 'scunthorpe', 'grimsby', 'shrewsbury', 'lowestoft', 'rugby', 'walsall', 'margate', 'blackburn', 'clacton', 'harrogate', 'scarborough', 'gloucester', 'south shields', 'great yarmouth', 'bootle', 'scunthorpe', 'grimsby', 'morecambe', 'thornton', 'bexhill', 'folkestone', 'warrington', 'workington', 'rhyl', 'southsea', 'loughborough', 'guildford', 'chatham', 'eastleigh', 'cheshunt', 'salisbury', 'yeovil', 'carlisle', 'greenock', 'hastings', 'harlow', 'woking', 'southall', 'acton', 'ealing', 'harrow', 'uxbridge', 'enfield', 'barnet', 'croydon', 'bromley', 'lewisham', 'greenwich', 'hackney', 'tower hamlets', 'newham', 'waltham forest', 'redbridge', 'havering', 'barking', 'dagenham', 'hillingdon', 'hounslow', 'richmond', 'kingston', 'merton', 'wandsworth', 'lambeth', 'southwark', 'camden', 'islington', 'haringey', 'kensington', 'chelsea', 'hammersmith', 'fulham', 'westminster', 'city of london']
            
            for city in potential_cities:
                if city in query_lower:
                    return f"🌍 **Travel Assistance for {city.title()}**\n\nI'd be happy to help with your travel query! However, I don't have specific information about {city.title()} in my current knowledge base.\n\n**What I can help you with instead:**\n• General travel advice and tips\n• Destination recommendations from our available dataset\n• Budget planning and travel tips\n• Best time to visit various destinations\n• Must-see attractions in destinations we have data for\n\n**Popular destinations I can help with:**\n• Sri Lanka - Cultural sites, beaches, wildlife\n• Various Asian destinations - Modern cities, cultural sites\n• European destinations - History, art, cuisine\n\nCould you try asking about a destination I have information about, or would you like general travel advice?"
        
        # If no specific match, provide general recommendations with all filters
        return self._format_general_recommendations(query_lower, conversation_filters)
    
    def _get_region_filter(self, query_lower: str) -> str:
        """Determine region filter from query."""
        # Asian countries
        asian_countries = [
            'china', 'japan', 'south korea', 'north korea', 'india', 'pakistan', 'bangladesh', 
            'sri lanka', 'nepal', 'bhutan', 'myanmar', 'thailand', 'laos', 'cambodia', 
            'vietnam', 'malaysia', 'singapore', 'indonesia', 'philippines', 'brunei', 
            'mongolia', 'kazakhstan', 'kyrgyzstan', 'tajikistan', 'turkmenistan', 'uzbekistan',
            'afghanistan', 'iran', 'iraq', 'syria', 'lebanon', 'jordan', 'israel', 
            'palestine', 'saudi arabia', 'yemen', 'oman', 'united arab emirates', 'qatar', 
            'bahrain', 'kuwait', 'turkey', 'russia', 'georgia', 'armenia', 'azerbaijan'
        ]
        
        # European countries
        european_countries = [
            'france', 'germany', 'italy', 'spain', 'portugal', 'netherlands', 'belgium', 
            'switzerland', 'austria', 'poland', 'czech republic', 'hungary', 'romania', 
            'bulgaria', 'greece', 'croatia', 'serbia', 'slovakia', 'slovenia', 'estonia', 
            'latvia', 'lithuania', 'finland', 'sweden', 'norway', 'denmark', 'iceland', 
            'ireland', 'united kingdom', 'ukraine', 'belarus', 'moldova', 'albania', 
            'bosnia', 'macedonia', 'montenegro', 'kosovo', 'luxembourg', 'malta', 'cyprus'
        ]
        
        # American countries
        american_countries = [
            'united states', 'canada', 'mexico', 'brazil', 'argentina', 'chile', 'peru', 
            'colombia', 'venezuela', 'ecuador', 'bolivia', 'paraguay', 'uruguay', 'guyana', 
            'suriname', 'french guiana', 'cuba', 'jamaica', 'haiti', 'dominican republic', 
            'puerto rico', 'trinidad and tobago', 'barbados', 'bahamas', 'belize', 'costa rica', 
            'panama', 'guatemala', 'honduras', 'el salvador', 'nicaragua'
        ]
        
        # Oceania countries
        oceania_countries = [
            'australia', 'new zealand', 'fiji', 'papua new guinea', 'solomon islands', 
            'vanuatu', 'samoa', 'tonga', 'micronesia', 'palau', 'marshall islands', 
            'kiribati', 'tuvalu', 'nauru', 'cook islands', 'french polynesia', 'new caledonia'
        ]
        
        # African countries
        african_countries = [
            'south africa', 'egypt', 'nigeria', 'kenya', 'morocco', 'algeria', 'tunisia', 
            'libya', 'sudan', 'ethiopia', 'ghana', 'tanzania', 'uganda', 'cameroon', 
            'ivory coast', 'madagascar', 'angola', 'mozambique', 'zambia', 'zimbabwe', 
            'botswana', 'namibia', 'senegal', 'mali', 'burkina faso', 'niger', 'chad', 
            'central african republic', 'democratic republic of congo', 'republic of congo', 
            'gabon', 'equatorial guinea', 'sao tome and principe', 'rwanda', 'burundi', 
            'somalia', 'djibouti', 'eritrea', 'mauritania', 'gambia', 'guinea-bissau', 
            'sierra leone', 'liberia', 'togo', 'benin', 'malawi', 'lesotho', 'swaziland', 
            'mauritius', 'seychelles', 'comoros', 'cape verde', 'sao tome'
        ]
        
        # Check for region keywords
        if any(keyword in query_lower for keyword in ['asia', 'asian', 'asiat']):
            return 'asia'
        elif any(keyword in query_lower for keyword in ['europe', 'european', 'europ']):
            return 'europe'
        elif any(keyword in query_lower for keyword in ['america', 'american', 'americas']):
            return 'america'
        elif any(keyword in query_lower for keyword in ['africa', 'african', 'afric']):
            return 'africa'
        elif any(keyword in query_lower for keyword in ['oceania', 'oceanic', 'australasia']):
            return 'oceania'
        
        # Check for specific countries
        if any(country in query_lower for country in asian_countries):
            return 'asia'
        elif any(country in query_lower for country in european_countries):
            return 'europe'
        elif any(country in query_lower for country in american_countries):
            return 'america'
        elif any(country in query_lower for country in african_countries):
            return 'africa'
        elif any(country in query_lower for country in oceania_countries):
            return 'oceania'
        
        return None
    
    def _extract_conversation_filters(self, query_lower: str, context: List[str]) -> dict:
        """Extract comprehensive filters from current query and conversation context."""
        filters = {
            'region': None,
            'category': None,
            'budget': None,
            'season': None,
            'travel_style': None,
            'interests': [],
            'group_type': None,
            'duration': None
        }
        
        # Combine current query with context for comprehensive analysis
        full_context = ' '.join(context + [query_lower])
        
        # Check if this is a general destination query (like "places to visit in X")
        is_general_destination_query = any(
            keyword in query_lower for keyword in [
                'places to visit', 'destinations in', 'things to see', 'attractions in',
                'where to go', 'best places', 'top places', 'must see', 'must visit'
            ]
        )
        
        # Extract region filter - prioritize current query over context
        current_region = self._get_region_filter(query_lower)
        context_region = self._get_region_filter(' '.join(context))
        
        # Check if current query is asking about a specific destination
        is_specific_destination_query = any(
            keyword in query_lower for keyword in [
                'tell me more about', 'more about', 'about', 'information about',
                'details about', 'tell me about', 'what is', 'where is'
            ]
        )
        
        # Only use context region if current query doesn't specify a region AND it's not a specific destination query
        if current_region:
            filters['region'] = current_region
        elif context_region and not is_specific_destination_query:
            filters['region'] = context_region
        
        # Extract category/type filter
        if any(keyword in full_context for keyword in ['beach', 'beaches', 'coastal', 'seaside', 'ocean', 'sea', 'surfing', 'diving', 'snorkeling']):
            filters['category'] = 'Beach'  # Special handling for beach destinations
        elif any(keyword in full_context for keyword in ['natural', 'nature', 'outdoor', 'wildlife', 'parks']):
            filters['category'] = 'Natural'
        elif any(keyword in full_context for keyword in ['cultural', 'culture', 'historical', 'heritage', 'museums', 'temples']):
            filters['category'] = 'Cultural'
        elif any(keyword in full_context for keyword in ['adventure', 'hiking', 'climbing', 'extreme', 'sports']):
            filters['category'] = 'Adventure'
        elif any(keyword in full_context for keyword in ['modern', 'city', 'urban', 'nightlife', 'shopping']):
            filters['category'] = 'Modern'
        
        # Extract budget filter - prioritize current query over context
        current_budget = None
        context_budget = None
        
        if any(keyword in query_lower for keyword in ['budget', 'cheap', 'affordable', 'low cost', 'economical']):
            current_budget = 'Budget'
        elif any(keyword in query_lower for keyword in ['luxury', 'expensive', 'high-end', 'premium', 'deluxe']):
            current_budget = 'Luxury'
        elif any(keyword in query_lower for keyword in ['mid-range', 'moderate', 'medium']):
            current_budget = 'Mid-range'
        
        if any(keyword in ' '.join(context) for keyword in ['budget', 'cheap', 'affordable', 'low cost', 'economical']):
            context_budget = 'Budget'
        elif any(keyword in ' '.join(context) for keyword in ['luxury', 'expensive', 'high-end', 'premium', 'deluxe']):
            context_budget = 'Luxury'
        elif any(keyword in ' '.join(context) for keyword in ['mid-range', 'moderate', 'medium']):
            context_budget = 'Mid-range'
        
        # Only use context budget if current query doesn't specify a budget AND it's not a general destination query
        if current_budget:
            filters['budget'] = current_budget
        elif context_budget and not is_general_destination_query:
            filters['budget'] = context_budget
        
        # Extract season filter - only for specific queries, not general destination queries
        if not is_general_destination_query:
            if any(keyword in full_context for keyword in ['summer', 'warm', 'hot']):
                filters['season'] = 'Summer'
            elif any(keyword in full_context for keyword in ['winter', 'cold', 'snow']):
                filters['season'] = 'Winter'
            elif any(keyword in full_context for keyword in ['spring', 'bloom', 'flowers']):
                filters['season'] = 'Spring'
            elif any(keyword in full_context for keyword in ['fall', 'autumn', 'harvest']):
                filters['season'] = 'Fall'
        
        # Extract travel style
        if any(keyword in full_context for keyword in ['family', 'kids', 'children']):
            filters['travel_style'] = 'Family'
        elif any(keyword in full_context for keyword in ['romantic', 'honeymoon', 'couple']):
            filters['travel_style'] = 'Romantic'
        elif any(keyword in full_context for keyword in ['solo', 'alone', 'backpacking']):
            filters['travel_style'] = 'Solo'
        elif any(keyword in full_context for keyword in ['group', 'friends', 'party']):
            filters['travel_style'] = 'Group'
        
        # Extract interests - only for specific queries, not general destination queries
        interests = []
        if not is_general_destination_query:
            if any(keyword in full_context for keyword in ['food', 'cuisine', 'restaurant', 'dining']):
                interests.append('Food')
            if any(keyword in full_context for keyword in ['beach', 'coastal', 'ocean', 'sea']):
                interests.append('Beach')
            if any(keyword in full_context for keyword in ['mountain', 'hiking', 'trekking']):
                interests.append('Mountains')
            if any(keyword in full_context for keyword in ['shopping', 'market', 'mall']):
                interests.append('Shopping')
            if any(keyword in full_context for keyword in ['nightlife', 'bars', 'clubs']):
                interests.append('Nightlife')
            if any(keyword in full_context for keyword in ['photography', 'instagram', 'scenic']):
                interests.append('Photography')
        filters['interests'] = interests
        
        # Extract group type
        if any(keyword in full_context for keyword in ['family', 'kids', 'children']):
            filters['group_type'] = 'Family'
        elif any(keyword in full_context for keyword in ['couple', 'romantic', 'honeymoon']):
            filters['group_type'] = 'Couple'
        elif any(keyword in full_context for keyword in ['solo', 'alone']):
            filters['group_type'] = 'Solo'
        elif any(keyword in full_context for keyword in ['friends', 'group']):
            filters['group_type'] = 'Friends'
        
        # Extract duration
        if any(keyword in full_context for keyword in ['weekend', '2 days', '3 days']):
            filters['duration'] = 'Short'
        elif any(keyword in full_context for keyword in ['week', '7 days', 'longer']):
            filters['duration'] = 'Medium'
        elif any(keyword in full_context for keyword in ['month', 'extended']):
            filters['duration'] = 'Long'
        
        return filters
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text for better readability."""
        import re
        
        if not text or text == 'nan' or text == 'None':
            return "No description available"
        
        # Convert to string and clean
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common concatenation issues - more aggressive approach
        # Split camelCase and ALLCAPS words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # word + Capitalized
        text = re.sub(r'([a-z])([A-Z][A-Z])', r'\1 \2', text)  # word + ALLCAPS
        
        # Split numbers from letters
        text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # letter + number
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)  # number + letter
        text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # number + lowercase
        
        # Fix specific common patterns
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase again
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase again
        
        # Handle common concatenated words (lowercase)
        common_words = [
            'beautiful', 'natural', 'waterfall', 'bath', 'water', 'nice', 'clam', 'place', 'cold',
            'nature', 'look', 'attention', 'please', 'think', 'protect', 'great', 'farm', 'coconut',
            'grove', 'spice', 'lake', 'bird', 'flower', 'tree', 'specimen', 'bar', 'entrance',
            'allow', 'least', 'two', 'hour', 'plant', 'ideal', 'visit', 'area', 'shop', 'park',
            'photographer', 'scene', 'take', 'capture', 'network', 'walking', 'trail', 'deep',
            'lush', 'observe', 'resident', 'migratory', 'every', 'single', 'time', 'travel',
            'love', 'sri', 'lanka', 'bit', 'close', 'gala', 'sharing', 'southern', 'coastline',
            'bundle', 'national', 'fewer', 'animal', 'safari', 'jeep', 'definitely', 'distance',
            'display', 'nearest', 'mile', 'kilometer', 'midigama', 'beach', 'left', 'surf', 'break'
        ]
        
        # Split concatenated words
        for word in common_words:
            # Look for word followed by another word (no space)
            pattern = f'({word})([a-z]{{2,}})'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        
        # Clean up spaces and special characters
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[∣|]', ' | ', text)  # Replace pipe symbols
        text = re.sub(r'[∗*]{2}', '**', text)  # Fix markdown bold
        text = re.sub(r'[−–]', '-', text)  # Normalize dashes
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)  # Final space cleanup
        
        return text.strip()
    
    def _format_sri_lanka_response(self, sri_lanka_destinations) -> str:
        """Format Sri Lanka destinations response using dedicated guide."""
        import pandas as pd
        
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
                # Clean up the attractions text
                attractions = self._clean_text(row['attractions'])
                
                # Truncate long descriptions
                if len(attractions) > 120:
                    attractions = attractions[:120] + "..."
                
                # Clean up budget and daily cost
                budget = self._clean_text(row['budget'])
                daily_cost = self._clean_text(row['daily_cost'])
                rating = self._clean_text(row['rating'])
                
                response += f"• **{row['destination']}** - {attractions}\n"
                response += f"  Budget: {budget} ({daily_cost}) | Rating: {rating}/5\n"
            
            response += "\n"
        
        response += "**Travel Tips:**\n"
        response += "• Best time: Year-round with dry season (Dec-Mar) preferred\n"
        response += "• Budget range: $25-80/day depending on style\n"
        response += "• Must-try: Local cuisine, tea plantations, cultural sites\n\n"
        response += "What type of experience interests you most?"
        
        return response
    
    def _format_country_response(self, country_destinations, country_name) -> str:
        """Format country destinations response."""
        import pandas as pd
        response = f"🌍 **Top Destinations in {country_name}**\n\n"
        response += f"Based on our travel dataset, here are the best places to visit in {country_name}:\n\n"
        
        # Filter out accommodation-related destinations and prioritize actual tourist destinations
        accommodation_keywords = [
            'hotel', 'resort', 'inn', 'club', 'villa', 'spa', 'apartment', 'guest house', 
            'hostel', 'lodge', 'camp', 'accommodation', 'stay', 'room', 'suite', 'bed',
            'motel', 'b&b', 'bed and breakfast', 'guesthouse', 'auberge', 'minshuku',
            'novotel', 'hyatt', 'hilton', 'marriott', 'sheraton', 'radisson', 'holiday inn',
            'ramada', 'best western', 'comfort inn', 'quality inn', 'days inn', 'super 8',
            'motel 6', 'red roof', 'la quinta', 'wyndham', 'courtyard', 'hampton', 'embassy',
            'doubletree', 'westin', 'ritz', 'four seasons', 'mandarin', 'peninsula', 'shangri-la'
        ]
        
        non_accommodation_destinations = country_destinations[
            ~country_destinations['destination'].str.contains('|'.join(accommodation_keywords), case=False, na=False)
        ]
        
        if len(non_accommodation_destinations) >= 6:
            # Use non-accommodation destinations if we have enough
            top_destinations = non_accommodation_destinations.nlargest(8, 'rating')
        elif len(non_accommodation_destinations) > 0:
            # Use available non-accommodation destinations plus some accommodation
            non_accommodation_destinations = non_accommodation_destinations.nlargest(8, 'rating')
            accommodation_destinations = country_destinations[
                country_destinations['destination'].str.contains('|'.join(accommodation_keywords), case=False, na=False)
            ].nlargest(2, 'rating')
            top_destinations = pd.concat([non_accommodation_destinations, accommodation_destinations])
        else:
            # If no non-accommodation destinations, provide a helpful message
            response += f"⚠️ **Note**: Our dataset for {country_name} primarily contains accommodation options rather than tourist attractions.\n\n"
            response += f"Here are some accommodation options in {country_name}:\n\n"
            top_destinations = country_destinations.nlargest(8, 'rating')
        
        for idx, row in top_destinations.iterrows():
            response += f"**{row['destination']}** ({row['category']})\n"
            # Handle NaN attractions
            attractions = str(row['attractions']) if pd.notna(row['attractions']) else "No description available"
            if len(attractions) > 100:
                attractions = attractions[:100] + "..."
            response += f"• {attractions}\n"
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
    
    def _format_general_recommendations(self, query_lower, conversation_filters=None) -> str:
        """Format general travel recommendations with comprehensive filtering."""
        import pandas as pd
        # Check if this is a follow-up to a previous recommendation
        if any(keyword in query_lower for keyword in ['natural', 'cultural', 'adventure', 'modern', 'beach', 'beaches', 'coastal', 'seaside', 'ocean', 'sea']):
            return self._format_category_specific_recommendations(query_lower, conversation_filters)
        
        # Apply comprehensive filters
        filtered_destinations = self._apply_comprehensive_filters(conversation_filters)
        
        if len(filtered_destinations) == 0:
            filter_summary = self._get_filter_summary(conversation_filters)
            return f"🌍 **No destinations found matching your criteria**\n\n{filter_summary}\n\nI don't have destinations matching all these criteria in our current dataset. Please try adjusting your preferences or ask about specific countries or regions we have data for."
        
        # Get top-rated destinations from filtered set, prioritizing non-accommodation destinations
        # First try to get non-accommodation destinations
        accommodation_keywords = [
            'hotel', 'resort', 'inn', 'club', 'villa', 'spa', 'apartment', 'guest house', 
            'hostel', 'lodge', 'camp', 'accommodation', 'stay', 'room', 'suite', 'bed',
            'motel', 'b&b', 'bed and breakfast', 'guesthouse', 'auberge', 'minshuku',
            'novotel', 'hyatt', 'hilton', 'marriott', 'sheraton', 'radisson', 'holiday inn',
            'ramada', 'best western', 'comfort inn', 'quality inn', 'days inn', 'super 8',
            'motel 6', 'red roof', 'la quinta', 'wyndham', 'courtyard', 'hampton', 'embassy',
            'doubletree', 'westin', 'ritz', 'four seasons', 'mandarin', 'peninsula', 'shangri-la'
        ]
        
        non_accommodation_destinations = filtered_destinations[
            ~filtered_destinations['destination'].str.contains('|'.join(accommodation_keywords), case=False, na=False)
        ]
        
        if len(non_accommodation_destinations) >= 8:
            # Use non-accommodation destinations if we have enough
            top_destinations = non_accommodation_destinations.nlargest(10, 'rating')
        elif len(non_accommodation_destinations) > 0:
            # Use available non-accommodation destinations plus some accommodation
            non_accommodation_destinations = non_accommodation_destinations.nlargest(8, 'rating')
            accommodation_destinations = filtered_destinations[
                filtered_destinations['destination'].str.contains('|'.join(accommodation_keywords), case=False, na=False)
            ].nlargest(2, 'rating')
            top_destinations = pd.concat([non_accommodation_destinations, accommodation_destinations])
        else:
            # If no non-accommodation destinations, provide a helpful message
            top_destinations = filtered_destinations.nlargest(10, 'rating')
        
        # Create filter summary for response
        filter_summary = self._get_filter_summary(conversation_filters)
        response = f"🌍 **Travel Recommendations**\n\n"
        
        # Add note if we're showing accommodation options
        if len(non_accommodation_destinations) == 0:
            response += f"⚠️ **Note**: Our dataset primarily contains accommodation options rather than tourist attractions.\n\n"
        if filter_summary:
            response += f"Based on your preferences: {filter_summary}\n\n"
        response += f"Here are the best matching destinations from our dataset:\n\n"
        
        for idx, row in top_destinations.iterrows():
            response += f"**{row['destination']}, {row['country']}** ({row['category']})\n"
            
            # Clean up all fields
            rating = self._clean_text(row['rating'])
            budget = self._clean_text(row['budget'])
            daily_cost = self._clean_text(row['daily_cost'])
            attractions = self._clean_text(row['attractions'])
            
            response += f"• Rating: {rating}/5 | Budget: {budget} ({daily_cost})\n"
            
            if len(attractions) > 100:
                attractions = attractions[:100] + "..."
            response += f"• {attractions}\n\n"
        
        response += "**What type of travel experience are you looking for?**\n"
        response += "• Cultural destinations\n"
        response += "• Natural attractions\n"
        response += "• Adventure activities\n"
        response += "• Modern cities\n\n"
        response += "Please let me know your preferences and I can provide more specific recommendations!"
        
        return response
    
    def _filter_destinations_by_region(self, region: str):
        """Filter destinations by region."""
        # Handle direct country names
        if region in self.travel_destinations['country'].values:
            return self.travel_destinations[self.travel_destinations['country'] == region]
        
        if region == 'asia':
            asian_countries = [
                'China', 'Japan', 'South Korea', 'North Korea', 'India', 'Pakistan', 'Bangladesh', 
                'Sri Lanka', 'Nepal', 'Bhutan', 'Myanmar', 'Thailand', 'Laos', 'Cambodia', 
                'Vietnam', 'Malaysia', 'Singapore', 'Indonesia', 'Philippines', 'Brunei', 
                'Mongolia', 'Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',
                'Afghanistan', 'Iran', 'Iraq', 'Syria', 'Lebanon', 'Jordan', 'Israel', 
                'Palestine', 'Saudi Arabia', 'Yemen', 'Oman', 'United Arab Emirates', 'Qatar', 
                'Bahrain', 'Kuwait', 'Turkey', 'Russia', 'Georgia', 'Armenia', 'Azerbaijan'
            ]
            return self.travel_destinations[
                self.travel_destinations['country'].isin(asian_countries)
            ]
        elif region == 'europe':
            european_countries = [
                'France', 'Germany', 'Italy', 'Spain', 'Portugal', 'Netherlands', 'Belgium', 
                'Switzerland', 'Austria', 'Poland', 'Czech Republic', 'Hungary', 'Romania', 
                'Bulgaria', 'Greece', 'Croatia', 'Serbia', 'Slovakia', 'Slovenia', 'Estonia', 
                'Latvia', 'Lithuania', 'Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland', 
                'Ireland', 'United Kingdom', 'Ukraine', 'Belarus', 'Moldova', 'Albania', 
                'Bosnia', 'Macedonia', 'Montenegro', 'Kosovo', 'Luxembourg', 'Malta', 'Cyprus'
            ]
            return self.travel_destinations[
                self.travel_destinations['country'].isin(european_countries)
            ]
        elif region == 'america':
            american_countries = [
                'United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Peru', 
                'Colombia', 'Venezuela', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 
                'Suriname', 'French Guiana', 'Cuba', 'Jamaica', 'Haiti', 'Dominican Republic', 
                'Puerto Rico', 'Trinidad and Tobago', 'Barbados', 'Bahamas', 'Belize', 'Costa Rica', 
                'Panama', 'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua'
            ]
            return self.travel_destinations[
                self.travel_destinations['country'].isin(american_countries)
            ]
        elif region == 'africa':
            african_countries = [
                'South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Morocco', 'Algeria', 'Tunisia', 
                'Libya', 'Sudan', 'Ethiopia', 'Ghana', 'Tanzania', 'Uganda', 'Cameroon', 
                'Ivory Coast', 'Madagascar', 'Angola', 'Mozambique', 'Zambia', 'Zimbabwe', 
                'Botswana', 'Namibia', 'Senegal', 'Mali', 'Burkina Faso', 'Niger', 'Chad', 
                'Central African Republic', 'Democratic Republic of Congo', 'Republic of Congo', 
                'Gabon', 'Equatorial Guinea', 'Sao Tome and Principe', 'Rwanda', 'Burundi', 
                'Somalia', 'Djibouti', 'Eritrea', 'Mauritania', 'Gambia', 'Guinea-Bissau', 
                'Sierra Leone', 'Liberia', 'Togo', 'Benin', 'Malawi', 'Lesotho', 'Swaziland', 
                'Mauritius', 'Seychelles', 'Comoros', 'Cape Verde', 'Sao Tome'
            ]
            return self.travel_destinations[
                self.travel_destinations['country'].isin(african_countries)
            ]
        elif region == 'oceania':
            oceania_countries = [
                'Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands', 
                'Vanuatu', 'Samoa', 'Tonga', 'Micronesia', 'Palau', 'Marshall Islands', 
                'Kiribati', 'Tuvalu', 'Nauru', 'Cook Islands', 'French Polynesia', 'New Caledonia'
            ]
            return self.travel_destinations[
                self.travel_destinations['country'].isin(oceania_countries)
            ]
        else:
            return self.travel_destinations
    
    def _apply_comprehensive_filters(self, conversation_filters):
        """Apply all conversation filters to destinations."""
        if not conversation_filters:
            return self.travel_destinations
        
        filtered_destinations = self.travel_destinations.copy()
        
        # Apply region filter
        if conversation_filters.get('region'):
            filtered_destinations = self._filter_destinations_by_region(conversation_filters['region'])
        
        # Apply category filter
        if conversation_filters.get('category'):
            if conversation_filters['category'] == 'Beach':
                # Special handling for beach destinations - prioritize actual beaches over hotels
                beach_keywords = ['beach', 'coastal', 'seaside', 'ocean', 'sea', 'surfing', 'diving', 'snorkeling']
                
                # First, try to find destinations that are clearly beaches (not hotels)
                beach_mask = (
                    # Destinations with "beach" in name but not "hotel", "resort", "inn", "club"
                    (
                        filtered_destinations['destination'].str.contains('beach', case=False, na=False) &
                        ~filtered_destinations['destination'].str.contains('hotel|resort|inn|club|villa|spa', case=False, na=False)
                    ) |
                    # Destinations with beach-related keywords in attractions
                    filtered_destinations['attractions'].str.contains('|'.join(beach_keywords), case=False, na=False) |
                    # Coastal/seaside destinations
                    filtered_destinations['destination'].str.contains('coastal|seaside|surf', case=False, na=False)
                )
                filtered_destinations = filtered_destinations[beach_mask]
            elif conversation_filters['category'] == 'Natural' and any(keyword in ' '.join(conversation_filters.get('interests', [])).lower() for keyword in ['mountain', 'mountains', 'hill', 'hills', 'peak', 'peaks']):
                # Special handling for mountain destinations - prioritize actual mountains over beaches
                mountain_keywords = ['mountain', 'hill', 'peak', 'summit', 'range', 'ridge', 'knuckles', 'adam', 'pidurutalagala', 'kirigalpotta', 'totapolakanda', 'hakgala', 'horton', 'world\'s end', 'ella rock', 'little adam\'s peak']
                
                # First, try to find destinations that are clearly mountains (not beaches or hotels)
                mountain_mask = (
                    # Destinations with mountain-related keywords in name
                    filtered_destinations['destination'].str.contains('|'.join(mountain_keywords), case=False, na=False) |
                    # Destinations with mountain-related keywords in attractions
                    filtered_destinations['attractions'].str.contains('|'.join(mountain_keywords), case=False, na=False) |
                    # Destinations with mountain-related keywords in description
                    filtered_destinations['description'].str.contains('|'.join(mountain_keywords), case=False, na=False)
                )
                filtered_destinations = filtered_destinations[mountain_mask]
            else:
                filtered_destinations = filtered_destinations[
                    filtered_destinations['category'] == conversation_filters['category']
                ]
        
        # Apply budget filter
        if conversation_filters.get('budget'):
            filtered_destinations = filtered_destinations[
                filtered_destinations['budget'] == conversation_filters['budget']
            ]
        
        # Apply season filter (best_time column)
        if conversation_filters.get('season'):
            season = conversation_filters['season']
            filtered_destinations = filtered_destinations[
                filtered_destinations['best_time'].str.contains(season, case=False, na=False)
            ]
        
        # Apply interests filter (search in attractions and description)
        if conversation_filters.get('interests'):
            interests = conversation_filters['interests']
            interest_mask = pd.Series([False] * len(filtered_destinations), index=filtered_destinations.index)
            
            for interest in interests:
                if interest == 'Food':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'food|cuisine|restaurant|dining|culinary', case=False, na=False
                    )
                elif interest == 'Beach':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'beach|coastal|ocean|sea|shore', case=False, na=False
                    )
                elif interest == 'Mountains':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'mountain|hiking|trekking|peak|summit', case=False, na=False
                    )
                elif interest == 'Shopping':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'shopping|market|mall|bazaar', case=False, na=False
                    )
                elif interest == 'Nightlife':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'nightlife|bars|clubs|entertainment', case=False, na=False
                    )
                elif interest == 'Photography':
                    interest_mask |= filtered_destinations['attractions'].str.contains(
                        'scenic|viewpoint|landscape|photography', case=False, na=False
                    )
            
            filtered_destinations = filtered_destinations[interest_mask]
        
        return filtered_destinations
    
    def _get_filter_summary(self, conversation_filters):
        """Create a human-readable summary of applied filters."""
        if not conversation_filters:
            return ""
        
        filters = []
        
        if conversation_filters.get('region'):
            filters.append(f"Region: {conversation_filters['region'].title()}")
        
        if conversation_filters.get('category'):
            filters.append(f"Type: {conversation_filters['category']}")
        
        if conversation_filters.get('budget'):
            filters.append(f"Budget: {conversation_filters['budget']}")
        
        if conversation_filters.get('season'):
            filters.append(f"Season: {conversation_filters['season']}")
        
        if conversation_filters.get('travel_style'):
            filters.append(f"Style: {conversation_filters['travel_style']}")
        
        if conversation_filters.get('interests'):
            interests_str = ', '.join(conversation_filters['interests'])
            filters.append(f"Interests: {interests_str}")
        
        if conversation_filters.get('group_type'):
            filters.append(f"Group: {conversation_filters['group_type']}")
        
        if conversation_filters.get('duration'):
            filters.append(f"Duration: {conversation_filters['duration']}")
        
        return " | ".join(filters) if filters else ""
    
    def _format_category_specific_recommendations(self, query_lower, conversation_filters=None) -> str:
        """Format recommendations based on specific category preference with comprehensive filtering."""
        # Determine category from query
        if any(keyword in query_lower for keyword in ['beach', 'beaches', 'coastal', 'seaside', 'ocean', 'sea', 'surfing', 'diving', 'snorkeling']):
            category = 'Beach'
            emoji = "🏖️"
        elif 'natural' in query_lower:
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
        
        # Update conversation filters with the category
        if conversation_filters is None:
            conversation_filters = {}
        conversation_filters['category'] = category
        
        # Apply comprehensive filters
        filtered_destinations = self._apply_comprehensive_filters(conversation_filters)
        
        # Get top destinations
        top_destinations = filtered_destinations.nlargest(8, 'rating')
        
        # Create filter summary for response
        filter_summary = self._get_filter_summary(conversation_filters)
        response = f"{emoji} **{category} Travel Destinations**\n\n"
        
        if filter_summary:
            response += f"Based on your preferences: {filter_summary}\n\n"
        
        response += f"Perfect! Here are the best {category.lower()} destinations from our dataset:\n\n"
        
        if len(top_destinations) > 0:
            for idx, row in top_destinations.iterrows():
                response += f"**{row['destination']}, {row['country']}**\n"
                
                # Clean up attractions text
                attractions = self._clean_text(row['attractions'])
                
                if len(attractions) > 100:
                    attractions = attractions[:100] + "..."
                
                response += f"• {attractions}\n"
                
                # Clean up budget, daily cost, rating, and best time
                budget = self._clean_text(row['budget'])
                daily_cost = self._clean_text(row['daily_cost'])
                rating = self._clean_text(row['rating'])
                best_time = self._clean_text(row['best_time'])
                
                response += f"• Budget: {budget} ({daily_cost}) | Rating: {rating}/5\n"
                response += f"• Best time: {best_time}\n\n"
        else:
            response += f"No {category.lower()} destinations found matching your criteria in our dataset.\n\n"
        
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
