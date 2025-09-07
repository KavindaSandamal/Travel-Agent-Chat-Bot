"""
Advanced AI Travel Advisor Chatbot - Web Interface
Complete Streamlit application demonstrating all Advanced AI concepts
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our Advanced AI modules
try:
    from src.nlp.preprocessing import AdvancedTextPreprocessor
    from src.embeddings.word_embeddings import WordEmbeddingGenerator
    from src.transformers.llm_integration import TravelTransformerPipeline
    from src.generative.generative_ai import TravelRAGSystem
    from src.training.few_shot_learning import TravelFewShotLearner
    from src.prompts.prompt_engineering import TravelPromptEngineer
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced modules not available: {e}")
    ADVANCED_MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Advanced AI Travel Advisor Chatbot",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Modern CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* General Text Color Fixes */
    .main p {
        color: #4a5568 !important;
    }
    
    .main div {
        color: #4a5568 !important;
    }
    
    .main span {
        color: #4a5568 !important;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #2d3748;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Chat Container */
    .chat-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Message Styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #38a169;
    }
    
    .user-message {
        background: #f7fafc;
        color: #2d3748;
        margin-left: 20%;
        border-left: 4px solid #4a5568;
        border: 1px solid #e2e8f0;
    }
    
    .bot-message {
        background: #f0fff4;
        color: #2d3748;
        margin-right: 20%;
        border-left: 4px solid #38a169;
        border: 1px solid #c6f6d5;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border-radius: 25px !important;
        border: 2px solid #e9ecef !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #38a169 !important;
        box-shadow: 0 0 0 3px rgba(56, 161, 105, 0.1) !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: #38a169;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(56, 161, 105, 0.2);
    }
    
    .stButton > button:hover {
        background: #2f855a;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        color: #2d3748;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .ai-concept {
        background: #f0fff4;
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #c6f6d5;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .response-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .response-box h3 {
        color: #2d3748 !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .response-box p {
        color: #4a5568 !important;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .response-box ul {
        color: #4a5568 !important;
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    .response-box li {
        color: #4a5568 !important;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .response-box strong {
        color: #2d3748 !important;
        font-weight: 600;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: #f7fafc;
    }
    
    /* Sidebar Text Styles */
    .css-1d391kg h3 {
        color: #2d3748 !important;
    }
    
    .css-1d391kg p {
        color: #4a5568 !important;
    }
    
    .css-1d391kg div {
        color: #4a5568 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: #38a169;
        border-radius: 10px;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        color: #6c757d;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        margin-left: 10px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        border-radius: 15px;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        border: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Quick Action Buttons */
    .quick-action {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .quick-action:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ai_components' not in st.session_state:
        st.session_state.ai_components = {}
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'travel_style': 'Cultural',
            'budget_range': 'Mid-range',
            'interests': ['History', 'Food', 'Art'],
            'previous_destinations': []
        }
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = {
            'queries_processed': 0,
            'response_time': [],
            'user_satisfaction': []
        }
    if 'quick_query' not in st.session_state:
        st.session_state.quick_query = None
    if 'is_typing' not in st.session_state:
        st.session_state.is_typing = False

def show_typing_indicator():
    """Show typing indicator."""
    st.markdown("""
    <div class="typing-indicator">
        🤖 TravelBot is thinking
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_ai_components():
    """Load AI components if available with caching."""
    if not ADVANCED_MODULES_AVAILABLE:
        return None
    
    try:
        # Check if components are already loaded and cached
        if 'ai_components' not in st.session_state or not st.session_state.ai_components:
            with st.spinner("🚀 Loading Advanced AI components (this may take a moment)..."):
                # Initialize components one by one with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                components = {}
                
                # 1. Text Preprocessor (fast)
                status_text.text("Loading NLP Preprocessor...")
                progress_bar.progress(10)
                components['text_preprocessor'] = AdvancedTextPreprocessor()
                
                # 2. Embedding Generator (medium)
                status_text.text("Loading Word Embeddings...")
                progress_bar.progress(25)
                components['embedding_generator'] = WordEmbeddingGenerator()
                
                # 3. RAG System (slow - loads enhanced datasets)
                status_text.text("Loading RAG System with enhanced travel data...")
                progress_bar.progress(50)
                rag_system = TravelRAGSystem()
                # Load enhanced dataset into RAG system
                rag_system.load_enhanced_dataset()
                components['rag_system'] = rag_system
                
                # Show dataset status
                travel_count = len(rag_system.travel_destinations) if rag_system.travel_destinations is not None else 0
                sri_lanka_count = len(rag_system.sri_lanka_guide) if rag_system.sri_lanka_guide is not None else 0
                status_text.text(f"✅ Loaded {travel_count} travel destinations and {sri_lanka_count} Sri Lanka destinations")
                
                # 4. Transformer Pipeline (slow - loads BERT)
                status_text.text("Loading Transformer Models...")
                progress_bar.progress(75)
                components['transformer_pipeline'] = TravelTransformerPipeline()
                
                # 5. Few-shot Learner (fast)
                status_text.text("Loading Few-shot Learning...")
                progress_bar.progress(90)
                components['few_shot_learner'] = TravelFewShotLearner()
                
                # 6. Prompt Engineer (fast)
                status_text.text("Loading Prompt Engineering...")
                progress_bar.progress(100)
                components['prompt_engineer'] = TravelPromptEngineer()
                
                # Cache the components
                st.session_state.ai_components = components
                st.session_state.components_loaded = True
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Advanced AI components loaded and cached successfully!")
        
        return st.session_state.ai_components
    except Exception as e:
        st.error(f"❌ Error loading AI components: {e}")
        return None

def create_sample_travel_data():
    """Create sample travel data for demonstration."""
    return {
        'destinations': [
            {
                'name': 'Paris',
                'country': 'France',
                'description': 'City of Light with world-class art, cuisine, and architecture',
                'best_time': 'Spring and Fall',
                'budget': 'Mid-range to High',
                'attractions': ['Eiffel Tower', 'Louvre Museum', 'Notre-Dame', 'Montmartre']
            },
            {
                'name': 'Tokyo',
                'country': 'Japan',
                'description': 'Blend of traditional culture and cutting-edge technology',
                'best_time': 'Spring and Fall',
                'budget': 'Mid-range',
                'attractions': ['Senso-ji Temple', 'Tokyo Skytree', 'Tsukiji Market', 'Shibuya Crossing']
            },
            {
                'name': 'London',
                'country': 'England',
                'description': 'Historic city with royal heritage and world-class museums',
                'best_time': 'Summer',
                'budget': 'High',
                'attractions': ['Buckingham Palace', 'British Museum', 'Tower of London', 'London Eye']
            },
            {
                'name': 'New York',
                'country': 'USA',
                'description': 'The city that never sleeps with Broadway, museums, and urban energy',
                'best_time': 'Spring and Fall',
                'budget': 'High',
                'attractions': ['Central Park', 'Broadway', 'Statue of Liberty', 'Times Square']
            },
            {
                'name': 'Sydney',
                'country': 'Australia',
                'description': 'Harbor city with beautiful beaches and iconic landmarks',
                'best_time': 'Spring and Fall',
                'budget': 'Mid-range to High',
                'attractions': ['Opera House', 'Harbour Bridge', 'Bondi Beach', 'Royal Botanic Gardens']
            }
        ]
    }

def process_query_simple(query: str) -> Dict[str, Any]:
    """Process query using simple rule-based approach."""
    travel_data = create_sample_travel_data()
    
    # Simple keyword matching
    query_lower = query.lower()
    
    # Find matching destinations
    matching_destinations = []
    for dest in travel_data['destinations']:
        if any(keyword in query_lower for keyword in [dest['name'].lower(), dest['country'].lower()]):
            matching_destinations.append(dest)
    
    # Generate response
    if matching_destinations:
        dest = matching_destinations[0]
        response = f"""**{dest['name']}, {dest['country']}**

{dest['description']}

**Best Time to Visit:** {dest['best_time']}
**Budget Level:** {dest['budget']}
**Top Attractions:** {', '.join(dest['attractions'])}

**Travel Tips:**
- Book accommodations in advance during peak season
- Consider getting a city pass for attractions
- Try local cuisine and cultural experiences
- Use public transportation for getting around

Is there anything specific about {dest['name']} you'd like to know more about?"""
    else:
        response = """I'd be happy to help with your travel questions! Here are some popular destinations I can tell you about:

**🌍 Top Destinations:**
- **Paris, France** - Art, culture, and cuisine
- **Tokyo, Japan** - Traditional and modern blend
- **London, England** - History and royal heritage
- **New York, USA** - Broadway and urban energy
- **Sydney, Australia** - Harbor city with beaches

**What I can help with:**
- Destination recommendations
- Travel planning and itineraries
- Best time to visit
- Budget considerations
- Local attractions and activities
- Cultural insights and tips

Please ask me about any of these destinations or let me know what type of travel experience you're looking for!"""
    
    return {
        'response': response,
        'intent': 'information',
        'confidence': 0.8,
        'destinations': [dest['name'] for dest in matching_destinations],
        'ai_techniques_used': ['Simple Rule-based', 'Keyword Matching']
    }

def process_query_advanced(query: str, components: Dict) -> Dict[str, Any]:
    """Process query using advanced AI components with conversation context."""
    try:
        # Get conversation context (last 3 messages)
        chat_history = st.session_state.get('chat_history', [])
        context_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history  # Last 3 exchanges
        
        # Step 1: NLP Preprocessing
        processed_query = components['text_preprocessor'].preprocess_pipeline(query)
        
        # Step 2: Transformer-based Processing
        intent_result = components['transformer_pipeline'].process_query(query)
        
        # Step 3: RAG-based Context Retrieval
        relevant_docs = components['rag_system'].retrieve_relevant_documents(query, k=3)
        
        # Step 4: Generate Response with conversation context
        if relevant_docs:
            # Extract conversation context for comprehensive filtering
            conversation_context = []
            if context_messages:
                # Extract user messages from recent conversation (last 3 messages only)
                recent_messages = context_messages[-6:]  # Last 6 messages (3 user + 3 bot pairs)
                for msg in recent_messages:
                    if msg['type'] == 'user':
                        conversation_context.append(msg['content'])
                
                # Only use context if the current query is very short AND seems like a follow-up
                # For longer, specific queries, don't use context
                if len(query.split()) > 2:  # If query has more than 2 words, it's likely specific
                    conversation_context = []
                elif len(query.split()) == 1:  # Single word queries
                    # Check if it's a category word that might be a follow-up
                    category_words = ['beaches', 'cultural', 'natural', 'adventure', 'modern', 'budget', 'luxury']
                    if query.lower() in category_words:
                        # For category words, be very conservative about maintaining context
                        # Only maintain context if the previous query was asking for the same category
                        if len(conversation_context) > 0:
                            prev_query = conversation_context[-1].lower()
                            # Only maintain context if the previous query was asking for the same category
                            # AND was very recent (within last 2 messages)
                            if (query.lower() in prev_query and 
                                len(conversation_context) <= 2 and
                                any(region in prev_query for region in ['asia', 'europe', 'america', 'africa'])):
                                # Keep context only if it's a clear continuation
                                pass
                            else:
                                conversation_context = []
            
            # Pass conversation context to the RAG system
            response = components['rag_system'].generate_response(query, relevant_docs, conversation_context)
        else:
            response = intent_result['response']
        
        # Step 5: Few-shot Learning Enhancement
        if intent_result['travel_info'].get('destinations'):
            destination = intent_result['travel_info']['destinations'][0]
            one_shot_response = components['few_shot_learner'].one_shot_learning(query, 'destination')
            response += f"\n\n**Personalized Insight:** {one_shot_response}"
        
        # Step 6: Advanced Prompt Engineering
        context = "\n".join([doc['content'] for doc in relevant_docs])
        dynamic_prompt = components['prompt_engineer'].create_dynamic_prompt("user", query, context)
        
        return {
            'response': response,
            'intent': intent_result['intent'],
            'confidence': intent_result['intent_confidence'],
            'entities': intent_result['entities'],
            'travel_info': intent_result['travel_info'],
            'relevant_documents': relevant_docs,
            'dynamic_prompt': dynamic_prompt,
            'conversation_context': len(context_messages),
            'ai_techniques_used': [
                'NLP Preprocessing',
                'Transformer Models',
                'RAG System',
                'Few-shot Learning',
                'Prompt Engineering',
                'Conversation Context'
            ]
        }
    except Exception as e:
        st.error(f"Error in advanced processing: {e}")
        return process_query_simple(query)

def display_ai_concepts():
    """Display Advanced AI concepts sidebar."""
    # Fast mode toggle
    st.sidebar.markdown("## ⚡ Performance Mode")
    fast_mode = st.sidebar.checkbox("Fast Mode (Simplified)", value=False, 
                                   help="Use simplified processing for faster responses")
    
    # Store fast mode in session state
    st.session_state.fast_mode = fast_mode
    
    # Display enhanced dataset information
    st.sidebar.markdown("## 📊 Enhanced Dataset")
    if 'ai_components' in st.session_state and st.session_state.ai_components:
        rag_system = st.session_state.ai_components.get('rag_system')
        if rag_system:
            travel_count = len(rag_system.travel_destinations) if rag_system.travel_destinations is not None else 0
            sri_lanka_count = len(rag_system.sri_lanka_guide) if rag_system.sri_lanka_guide is not None else 0
            st.sidebar.success(f"🌍 {travel_count:,} Travel Destinations")
            st.sidebar.success(f"🇱🇰 {sri_lanka_count:,} Sri Lanka Destinations")
        else:
            st.sidebar.info("📊 Loading enhanced datasets...")
    else:
        st.sidebar.info("📊 Enhanced datasets will load when you start chatting")
    
    st.sidebar.markdown("## 🎓 Advanced AI Concepts")
    
    concepts = [
        ("🧠 NLP", "Text preprocessing, tokenization, sentiment analysis, NER"),
        ("🔤 Embeddings", "Word2Vec, TF-IDF, Sentence Transformers"),
        ("🤖 Transformers", "BERT, GPT, Custom architectures"),
        ("🎨 Generative AI", "Autoencoders, RAG, GANs"),
        ("🎯 Few-shot Learning", "Prototypical Networks, MAML"),
        ("📝 Prompt Engineering", "Templates, few-shot, chain-of-thought"),
        ("🔧 MLOps", "Training, deployment, monitoring")
    ]
    
    for concept, description in concepts:
        with st.sidebar.expander(concept):
            st.write(description)

def display_system_metrics():
    """Display system performance metrics."""
    st.sidebar.markdown("## 📊 System Metrics")
    
    metrics = st.session_state.system_metrics
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Queries Processed", metrics['queries_processed'])
    
    with col2:
        avg_response_time = np.mean(metrics['response_time']) if metrics['response_time'] else 0
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    if metrics['user_satisfaction']:
        avg_satisfaction = np.mean(metrics['user_satisfaction'])
        st.sidebar.metric("User Satisfaction", f"{avg_satisfaction:.1f}/5")

def display_user_profile():
    """Display and edit user profile."""
    st.sidebar.markdown("## 👤 User Profile")
    
    with st.sidebar.expander("Edit Profile"):
        travel_style = st.selectbox(
            "Travel Style",
            ["Cultural", "Adventure", "Relaxation", "Food", "History", "Nature"],
            index=["Cultural", "Adventure", "Relaxation", "Food", "History", "Nature"].index(st.session_state.user_profile['travel_style'])
        )
        
        budget_range = st.selectbox(
            "Budget Range",
            ["Budget", "Mid-range", "Luxury"],
            index=["Budget", "Mid-range", "Luxury"].index(st.session_state.user_profile['budget_range'])
        )
        
        interests = st.multiselect(
            "Interests",
            ["History", "Food", "Art", "Nature", "Adventure", "Culture", "Shopping", "Nightlife"],
            default=st.session_state.user_profile['interests']
        )
        
        if st.button("Update Profile"):
            st.session_state.user_profile.update({
                'travel_style': travel_style,
                'budget_range': budget_range,
                'interests': interests
            })
            st.success("Profile updated!")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Beautiful Header
    st.markdown('<h1 class="main-header">🌍 Advanced AI Travel Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by 6,649+ Destinations</p>', unsafe_allow_html=True)
    
    # Quick Action Buttons
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏝️ Best Places Sri Lanka", key="quick_sri_lanka"):
            st.session_state.quick_query = "best places to visit in Sri Lanka"
            st.rerun()
    
    with col2:
        if st.button("🌍 Top Destinations", key="quick_destinations"):
            st.session_state.quick_query = "top travel destinations in the world"
            st.rerun()
    
    with col3:
        if st.button("💰 Budget Travel", key="quick_budget"):
            st.session_state.quick_query = "budget travel destinations"
            st.rerun()
    
    with col4:
        if st.button("🏛️ Cultural Sites", key="quick_cultural"):
            st.session_state.quick_query = "best cultural destinations"
            st.rerun()
    
    st.markdown("---")
    
    # Welcome message for new users
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="response-box">
            <h3>🌟 Welcome to Advanced AI Travel Advisor!</h3>
            <p>I'm your intelligent travel companion powered by advanced AI technologies including:</p>
            <ul>
                <li>🧠 Natural Language Processing (NLP)</li>
                <li>🔤 Word Embedding Methods</li>
                <li>🤖 Transformer-based Models & LLMs</li>
                <li>🎨 Generative AI (RAG, Autoencoders, GANs)</li>
                <li>🎯 Few-shot Learning</li>
                <li>📝 Advanced Prompt Engineering</li>
            </ul>
            <p><strong>I have access to 6,649+ travel destinations and 2,435 Sri Lanka destinations!</strong></p>
            <p>Try asking me about destinations, comparing places, or getting travel recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    display_ai_concepts()
    display_system_metrics()
    display_user_profile()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💬 Chat with TravelBot</h2>', unsafe_allow_html=True)
        
        # Enhanced chat input
        st.markdown("### 💬 Ask TravelBot Anything")
        user_input = st.text_input(
            "Ask TravelBot",
            placeholder="🌍 Ask about destinations, compare places, get travel tips...",
            key="user_input",
            help="Powered by 6,649+ destinations and advanced AI",
            label_visibility="collapsed"
        )
        
        # Handle quick query
        if st.session_state.quick_query:
            user_input = st.session_state.quick_query
            st.session_state.quick_query = None
            # Automatically process the quick query
            if user_input and user_input.strip():
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
                
                # Process the query
                with st.spinner("🤖 TravelBot is thinking..."):
                    st.session_state.is_typing = True
                    components = load_ai_components()
                    result = process_query_advanced(user_input, components)
                    st.session_state.is_typing = False
                    
                    # Extract the actual response text from the result dictionary
                    response = result.get('response', 'Sorry, I could not process your request.')
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response,
                        'timestamp': datetime.now().strftime("%H:%M")
                    })
                
                # Clear the input and rerun
                st.rerun()
        
        # Enhanced send button
        if st.button("🚀 Send Message", type="primary"):
            if user_input and user_input.strip():
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
                
                # Process query
                start_time = time.time()
                
                # Check if fast mode is enabled
                fast_mode = st.session_state.get('fast_mode', False)
                
                if fast_mode:
                    # Use simple processing for fast responses
                    result = process_query_simple(user_input)
                else:
                    # Load AI components (cached after first load)
                    components = load_ai_components()
                    
                    if components and ADVANCED_MODULES_AVAILABLE:
                        result = process_query_advanced(user_input, components)
                    else:
                        result = process_query_simple(user_input)
                
                response_time = time.time() - start_time
                
                # Extract response text and add to chat history
                response_text = result.get('response', 'Sorry, I could not process your request.')
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': response_text,
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'metadata': {
                        'intent': result.get('intent', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'ai_techniques': result.get('ai_techniques_used', []),
                        'response_time': response_time
                    }
                })
                
                # Update metrics
                st.session_state.system_metrics['queries_processed'] += 1
                st.session_state.system_metrics['response_time'].append(response_time)
                
                # Clear input by rerunning
                st.rerun()
        
        # Display chat history with beautiful styling
        for message in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 TravelBot:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show metadata if available
                if 'metadata' in message:
                    metadata = message['metadata']
                    with st.expander("🔍 AI Analysis Details", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**🎯 Intent:** {metadata['intent']}")
                            st.markdown(f"**📊 Confidence:** {metadata['confidence']:.3f}")
                        with col_b:
                            st.markdown(f"**⏱️ Response Time:** {metadata['response_time']:.2f}s")
                            st.markdown(f"**🧠 AI Techniques:** {', '.join(metadata['ai_techniques'])}")
        
        # Example questions
        st.markdown('<h3 class="sub-header">💡 Example Questions</h3>', unsafe_allow_html=True)
        
        example_questions = [
            "Tell me about Paris",
            "Compare Tokyo and London for families",
            "What's the best time to visit Sydney?",
            "Plan a 5-day trip to Rome",
            "Recommend budget destinations in Europe"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}"):
                    # Add the question to chat history directly
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'content': question,
                        'timestamp': datetime.now().strftime("%H:%M")
                    })
                    
                    # Process the example question
                    start_time = time.time()
                    
                    # Check if fast mode is enabled
                    fast_mode = st.session_state.get('fast_mode', False)
                    
                    if fast_mode:
                        # Use simple processing for fast responses
                        result = process_query_simple(question)
                    else:
                        # Load AI components (cached after first load)
                        components = load_ai_components()
                        
                        if components and ADVANCED_MODULES_AVAILABLE:
                            result = process_query_advanced(question, components)
                        else:
                            result = process_query_simple(question)
                    
                    response_time = time.time() - start_time
                    
                    # Extract response text and add to chat history
                    response_text = result.get('response', 'Sorry, I could not process your request.')
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': response_text,
                        'timestamp': datetime.now().strftime("%H:%M"),
                        'metadata': {
                            'intent': result.get('intent', 'unknown'),
                            'confidence': result.get('confidence', 0.0),
                            'ai_techniques': result.get('ai_techniques_used', []),
                            'response_time': response_time
                        }
                    })
                    
                    # Update metrics
                    st.session_state.system_metrics['queries_processed'] += 1
                    st.session_state.system_metrics['response_time'].append(response_time)
                    
                    st.rerun()
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 System Status</h2>', unsafe_allow_html=True)
        
        # AI Components Status
        st.markdown("**🤖 AI Components:**")
        if ADVANCED_MODULES_AVAILABLE:
            st.success("✅ Advanced AI modules loaded")
            st.success("✅ NLP Preprocessing")
            st.success("✅ Word Embeddings")
            st.success("✅ Transformer Models")
            st.success("✅ Generative AI")
            st.success("✅ Few-shot Learning")
            st.success("✅ Prompt Engineering")
        else:
            st.warning("⚠️ Using simplified mode")
            st.info("ℹ️ Advanced modules not available")
        
        # Performance Metrics
        st.markdown("**📈 Performance:**")
        metrics = st.session_state.system_metrics
        
        if metrics['queries_processed'] > 0:
            avg_response_time = np.mean(metrics['response_time'])
            st.metric("Average Response Time", f"{avg_response_time:.2f}s")
            st.metric("Total Queries", metrics['queries_processed'])
        else:
            st.info("No queries processed yet")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.system_metrics = {
                'queries_processed': 0,
                'response_time': [],
                'user_satisfaction': []
            }
            st.rerun()
        
        # Export chat history
        if st.session_state.chat_history:
            chat_data = []
            for msg in st.session_state.chat_history:
                # Clean metadata to ensure JSON serialization
                metadata = msg.get('metadata', {})
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, datetime):
                        clean_metadata[key] = value.isoformat()
                    else:
                        clean_metadata[key] = value
                
                chat_data.append({
                    'timestamp': msg['timestamp'],  # Already a string from strftime
                    'type': msg['type'],
                    'content': msg['content'],
                    'metadata': clean_metadata
                })
            
            json_data = json.dumps(chat_data, indent=2)
            st.download_button(
                label="📥 Download Chat History",
                data=json_data,
                file_name=f"travel_chatbot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🎓 Advanced AI Travel Advisor Chatbot </p>
        <p>Demonstrating: NLP • Embeddings • Transformers • Generative AI • Few-shot Learning • Prompt Engineering • MLOps</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
