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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ai-concept {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e8b57;
        margin: 0.5rem 0;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
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
                
                # 3. RAG System (slow - loads datasets)
                status_text.text("Loading RAG System with travel data...")
                progress_bar.progress(50)
                components['rag_system'] = TravelRAGSystem()
                
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
            # Add conversation context to the query for better responses
            contextual_query = query
            if context_messages:
                # Extract key topics from recent conversation
                recent_topics = []
                for msg in context_messages:
                    if msg['type'] == 'user':
                        recent_topics.append(msg['content'])
                if recent_topics:
                    contextual_query = f"Previous context: {' '.join(recent_topics[-2:])}. Current query: {query}"
            
            response = components['rag_system'].generate_response(contextual_query, relevant_docs)
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
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Advanced AI Travel Advisor Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">🎓 Academic Assignment - Advanced AI Course</p>', unsafe_allow_html=True)
    
    # Sidebar
    display_ai_concepts()
    display_system_metrics()
    display_user_profile()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💬 Chat with TravelBot</h2>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            "Ask me anything about travel!",
            placeholder="e.g., 'Tell me about Paris' or 'Compare Tokyo and London'",
            key="user_input"
        )
        
        # Send button
        if st.button("Send", type="primary"):
            if user_input and user_input.strip():
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
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
                
                # Add response to chat history
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': result['response'],
                    'timestamp': datetime.now(),
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
        
        # Display chat history
        for message in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message['type'] == 'user':
                st.markdown(f"**👤 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 TravelBot:** {message['content']}")
                
                # Show metadata if available
                if 'metadata' in message:
                    metadata = message['metadata']
                    with st.expander("🔍 AI Analysis Details"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Intent:** {metadata['intent']}")
                            st.write(f"**Confidence:** {metadata['confidence']:.3f}")
                        with col_b:
                            st.write(f"**Response Time:** {metadata['response_time']:.2f}s")
                            st.write(f"**AI Techniques:** {', '.join(metadata['ai_techniques'])}")
        
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
                        'timestamp': datetime.now()
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
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': result['response'],
                        'timestamp': datetime.now(),
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
                chat_data.append({
                    'timestamp': msg['timestamp'].isoformat(),
                    'type': msg['type'],
                    'content': msg['content'],
                    'metadata': msg.get('metadata', {})
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
        <p>🎓 Advanced AI Travel Advisor Chatbot - Academic Assignment</p>
        <p>Demonstrating: NLP • Embeddings • Transformers • Generative AI • Few-shot Learning • Prompt Engineering • MLOps</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
