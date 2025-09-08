"""
Transformer-based Models and LLMs Integration for Travel Advisor Chatbot
Demonstrates BERT, GPT, and custom transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel,
    pipeline, set_seed
)
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class TravelBERTClassifier(nn.Module):
    """
    Custom BERT-based classifier for travel intent classification.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 10):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name (str): BERT model name
            num_classes (int): Number of classification classes
        """
        super(TravelBERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            torch.Tensor: Classification logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class TravelGPTGenerator:
    """
    GPT-based text generator for travel responses.
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize the GPT generator.
        
        Args:
            model_name (str): GPT model name
        """
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set seed for reproducibility
        set_seed(42)
    
    def generate_response(self, prompt: str, max_length: int = 150, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate travel response using GPT.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def generate_travel_recommendation(self, destination: str, preferences: str = "") -> str:
        """
        Generate travel recommendation for a destination.
        
        Args:
            destination (str): Destination name
            preferences (str): User preferences
            
        Returns:
            str: Generated recommendation
        """
        prompt = f"Travel recommendation for {destination}"
        if preferences:
            prompt += f" with preferences: {preferences}"
        prompt += ":"
        
        return self.generate_response(prompt, max_length=200)
    
    def generate_comparison(self, destination1: str, destination2: str) -> str:
        """
        Generate comparison between two destinations.
        
        Args:
            destination1 (str): First destination
            destination2 (str): Second destination
            
        Returns:
            str: Generated comparison
        """
        prompt = f"Compare {destination1} vs {destination2} for travel:"
        return self.generate_response(prompt, max_length=250)

class TravelIntentClassifier:
    """
    BERT-based intent classifier for travel queries.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the intent classifier.
        
        Args:
            model_name (str): BERT model name
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Define travel intents
        self.intents = [
            'recommendation', 'information', 'comparison', 'planning',
            'booking', 'weather', 'safety', 'budget', 'activities', 'general'
        ]
        
        self.intent_to_id = {intent: i for i, intent in enumerate(self.intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        # Initialize model
        self.model = TravelBERTClassifier(model_name, len(self.intents))
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of a travel query.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[str, float]: Predicted intent and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(**inputs)
            probabilities = F.softmax(logits, dim=1)
            predicted_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_intent = self.id_to_intent[predicted_id]
        return predicted_intent, confidence
    
    def batch_classify(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify intents for multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Tuple[str, float]]: List of (intent, confidence) tuples
        """
        results = []
        for text in texts:
            intent, confidence = self.classify_intent(text)
            results.append((intent, confidence))
        return results

class TravelEntityExtractor:
    """
    BERT-based named entity recognition for travel entities.
    """
    
    def __init__(self, model_name: str = 'dbmdz/bert-large-cased-finetuned-conll03-english'):
        """
        Initialize the entity extractor.
        
        Args:
            model_name (str): BERT NER model name
        """
        self.model_name = model_name
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple"
        )
        
        # Travel-specific entity types
        self.travel_entities = {
            'LOC': 'Location',
            'GPE': 'Geopolitical Entity',
            'ORG': 'Organization',
            'MONEY': 'Money',
            'DATE': 'Date',
            'TIME': 'Time'
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of extracted entities
        """
        entities = self.ner_pipeline(text)
        
        # Filter for travel-relevant entities
        travel_entities = []
        for entity in entities:
            if entity['entity_group'] in self.travel_entities:
                travel_entities.append({
                    'text': entity['word'],
                    'label': self.travel_entities[entity['entity_group']],
                    'confidence': entity['score']
                })
        
        return travel_entities
    
    def extract_travel_info(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured travel information from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Structured travel information
        """
        entities = self.extract_entities(text)
        
        travel_info = {
            'destinations': [],
            'organizations': [],
            'money': [],
            'dates': [],
            'times': []
        }
        
        for entity in entities:
            label = entity['label']
            text_entity = entity['text']
            
            if label == 'Location' or label == 'Geopolitical Entity':
                travel_info['destinations'].append(text_entity)
            elif label == 'Organization':
                travel_info['organizations'].append(text_entity)
            elif label == 'Money':
                travel_info['money'].append(text_entity)
            elif label == 'Date':
                travel_info['dates'].append(text_entity)
            elif label == 'Time':
                travel_info['times'].append(text_entity)
        
        return travel_info

class TravelTransformerPipeline:
    """
    Complete transformer pipeline for travel chatbot.
    """
    
    def __init__(self):
        """Initialize the transformer pipeline."""
        self.intent_classifier = TravelIntentClassifier()
        self.entity_extractor = TravelEntityExtractor()
        self.gpt_generator = TravelGPTGenerator()
        
        # Travel knowledge base
        self.travel_knowledge = {
            'paris': {
                'description': 'City of Light with world-class museums and cuisine',
                'attractions': ['Eiffel Tower', 'Louvre', 'Notre-Dame'],
                'best_time': 'Spring and Fall',
                'budget': 'Mid-range to High'
            },
            'tokyo': {
                'description': 'Blend of traditional and modern culture',
                'attractions': ['Senso-ji Temple', 'Tokyo Skytree', 'Tsukiji Market'],
                'best_time': 'Spring and Fall',
                'budget': 'Mid-range'
            },
            'london': {
                'description': 'Historic city with royal heritage',
                'attractions': ['Buckingham Palace', 'British Museum', 'Tower of London'],
                'best_time': 'Summer',
                'budget': 'High'
            }
        }
    
    def process_query(self, query: str) -> Dict[str, Union[str, List, float]]:
        """
        Process a travel query using the complete pipeline.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Union[str, List, float]]: Processed response
        """
        # Step 1: Classify intent
        intent, intent_confidence = self.intent_classifier.classify_intent(query)
        
        # Step 2: Extract entities
        entities = self.entity_extractor.extract_entities(query)
        travel_info = self.entity_extractor.extract_travel_info(query)
        
        # Step 3: Generate response based on intent
        response = self._generate_contextual_response(query, intent, travel_info)
        
        return {
            'query': query,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'entities': entities,
            'travel_info': travel_info,
            'response': response
        }
    
    def _generate_contextual_response(self, query: str, intent: str, travel_info: Dict) -> str:
        """
        Generate contextual response based on intent and extracted information.
        
        Args:
            query (str): Original query
            intent (str): Classified intent
            travel_info (Dict): Extracted travel information
            
        Returns:
            str: Generated response
        """
        destinations = travel_info.get('destinations', [])
        
        if intent == 'recommendation':
            if destinations:
                destination = destinations[0].lower()
                if destination in self.travel_knowledge:
                    knowledge = self.travel_knowledge[destination]
                    prompt = f"Recommend {destination} for travel: {knowledge['description']}"
                else:
                    prompt = f"Recommend {destination} for travel"
            else:
                prompt = "Recommend travel destinations"
            
            return self.gpt_generator.generate_response(prompt, max_length=200)
        
        elif intent == 'comparison':
            if len(destinations) >= 2:
                return self.gpt_generator.generate_comparison(destinations[0], destinations[1])
            else:
                return "Please specify two destinations to compare."
        
        elif intent == 'information':
            if destinations:
                destination = destinations[0].lower()
                if destination in self.travel_knowledge:
                    knowledge = self.travel_knowledge[destination]
                    return f"{destination.title()}: {knowledge['description']}. Attractions: {', '.join(knowledge['attractions'])}. Best time: {knowledge['best_time']}. Budget: {knowledge['budget']}."
                else:
                    prompt = f"Provide information about {destination}"
                    return self.gpt_generator.generate_response(prompt, max_length=150)
            else:
                return "Please specify a destination for information."
        
        else:
            # General response
            return self.gpt_generator.generate_response(query, max_length=150)
    
    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn-style training.
        
        Args:
            X: Training data (not used in this implementation)
            y: Target labels (not used in this implementation)
        """
        # This is a pre-trained pipeline, so no actual training is needed
        # Just initialize components if needed
        print("🤖 TravelTransformerPipeline: Using pre-trained components")
        return self
    
    def predict(self, X):
        """
        Predict method for compatibility with sklearn-style evaluation.
        
        Args:
            X: Input data (list of queries)
            
        Returns:
            List of predictions
        """
        predictions = []
        for query in X:
            result = self.process_query(query)
            # Return the intent as the prediction
            predictions.append(result['intent'])
        return predictions
    
    def score(self, X, y):
        """
        Score method for compatibility with sklearn-style evaluation.
        
        Args:
            X: Input data (list of queries)
            y: True labels (list of intents)
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def main():
    """Demonstrate transformer models and LLMs."""
    print("🤖 Transformer-based Models and LLMs Demo")
    print("=" * 50)
    
    # Initialize the pipeline
    pipeline = TravelTransformerPipeline()
    
    # Sample travel queries
    test_queries = [
        "I want to visit Paris next spring",
        "Compare Tokyo and London for a family vacation",
        "Tell me about the weather in Sydney",
        "What are the best restaurants in Rome?",
        "Plan a budget trip to Barcelona"
    ]
    
    print("🔍 Processing Travel Queries:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Process query
        result = pipeline.process_query(query)
        
        print(f"   Intent: {result['intent']} (confidence: {result['intent_confidence']:.3f})")
        print(f"   Entities: {[e['text'] for e in result['entities']]}")
        print(f"   Destinations: {result['travel_info']['destinations']}")
        print(f"   Response: {result['response'][:100]}...")
    
    # Demonstrate individual components
    print("\n🧠 Individual Component Demos:")
    print("-" * 30)
    
    # Intent Classification
    print("\n1. Intent Classification:")
    intent_classifier = TravelIntentClassifier()
    test_texts = [
        "Recommend places to visit in Europe",
        "What's the weather like in Tokyo?",
        "Compare Paris and London",
        "Book a hotel in New York"
    ]
    
    for text in test_texts:
        intent, confidence = intent_classifier.classify_intent(text)
        print(f"   '{text}' -> {intent} ({confidence:.3f})")
    
    # Entity Extraction
    print("\n2. Entity Extraction:")
    entity_extractor = TravelEntityExtractor()
    test_text = "I want to visit Paris in March 2024 with a budget of $2000"
    entities = entity_extractor.extract_entities(test_text)
    print(f"   Text: {test_text}")
    print(f"   Entities: {entities}")
    
    # GPT Generation
    print("\n3. GPT Text Generation:")
    gpt_generator = TravelGPTGenerator()
    prompt = "Travel recommendation for Paris:"
    response = gpt_generator.generate_response(prompt, max_length=100)
    print(f"   Prompt: {prompt}")
    print(f"   Response: {response}")
    
    print("\n✅ Transformer Models and LLMs Demo Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Custom BERT classifier for intent classification")
    print("  - BERT-based named entity recognition")
    print("  - GPT-2 text generation")
    print("  - Complete transformer pipeline")
    print("  - Travel-specific knowledge integration")

if __name__ == "__main__":
    main()
