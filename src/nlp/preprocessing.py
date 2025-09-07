"""
Advanced NLP Text Pre/Post Processing for Travel Advisor Chatbot
Demonstrates comprehensive text processing techniques
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
try:
    import spacy
except ImportError:
    spacy = None
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing class demonstrating comprehensive NLP techniques.
    """
    
    def __init__(self, language='english'):
        """Initialize the advanced preprocessor."""
        self.language = language
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize spaCy (optional for Windows compatibility)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            print("spaCy not available. Using NLTK-only mode.")
            self.nlp = None
        
        # Travel-specific patterns
        self.travel_patterns = {
            'price': r'\$[\d,]+(?:\.\d{2})?',
            'rating': r'\b\d+\.?\d*\s*(?:star|stars|/5|/10)\b',
            'date': r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'time': r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'currency': r'\b(?:USD|EUR|GBP|JPY|CAD|AUD)\b'
        }
        
        # Travel-specific stopwords
        self.travel_stopwords = {
            'hotel', 'place', 'location', 'area', 'city', 'town', 'visit', 'travel',
            'trip', 'vacation', 'holiday', 'destination', 'stay', 'stayed'
        }
        self.stop_words.update(self.travel_stopwords)
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning with multiple techniques.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for travel-specific punctuation
        text = re.sub(r'[^\w\s$.,!?]', '', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str, method: str = 'nltk') -> List[str]:
        """
        Advanced tokenization with multiple methods.
        
        Args:
            text (str): Text to tokenize
            method (str): Tokenization method ('nltk', 'spacy', 'simple')
            
        Returns:
            List[str]: Tokenized text
        """
        if method == 'nltk':
            return word_tokenize(text)
        elif method == 'spacy' and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return text.split()
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using WordNet lemmatizer.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        lemmatized = []
        for token in tokens:
            lemmatized.append(self.lemmatizer.lemmatize(token))
        return lemmatized
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using Porter stemmer.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        stemmed = []
        for token in tokens:
            stemmed.append(self.stemmer.stem(token))
        return stemmed
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def extract_travel_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract travel-specific entities from text.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Dict[str, List[str]]: Extracted entities by category
        """
        entities = {
            'prices': [],
            'ratings': [],
            'dates': [],
            'times': [],
            'locations': [],
            'currencies': []
        }
        
        # Extract using regex patterns
        for entity_type, pattern in self.travel_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type + 's'] = matches
        
        # Extract using spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC', 'ORG']:  # Geopolitical, Location, Organization
                    entities['locations'].append(ent.text)
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        return self.sentiment_analyzer.polarity_scores(text)
    
    def extract_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract part-of-speech tags.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[Tuple[str, str]]: Word-POS tag pairs
        """
        tokens = word_tokenize(text)
        return pos_tag(tokens)
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using NLTK.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[Tuple[str, str]]: Entity-label pairs
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entities.append((chunk.leaves()[0][0], chunk.label()))
        
        return entities
    
    def preprocess_pipeline(self, text: str, 
                          tokenize: bool = True,
                          lemmatize: bool = True,
                          remove_stops: bool = True,
                          extract_entities: bool = True,
                          analyze_sentiment: bool = True) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw text
            tokenize (bool): Whether to tokenize
            lemmatize (bool): Whether to lemmatize
            remove_stops (bool): Whether to remove stopwords
            extract_entities (bool): Whether to extract entities
            analyze_sentiment (bool): Whether to analyze sentiment
            
        Returns:
            Dict: Processed text data
        """
        result = {
            'original_text': text,
            'cleaned_text': self.clean_text(text),
            'tokens': [],
            'lemmatized_tokens': [],
            'entities': {},
            'sentiment': {},
            'pos_tags': [],
            'named_entities': []
        }
        
        # Clean text
        cleaned_text = result['cleaned_text']
        
        # Tokenize
        if tokenize:
            tokens = self.tokenize_text(cleaned_text)
            result['tokens'] = tokens
            
            # Remove stopwords
            if remove_stops:
                tokens = self.remove_stopwords(tokens)
                result['tokens'] = tokens
            
            # Lemmatize
            if lemmatize:
                lemmatized = self.lemmatize_tokens(tokens)
                result['lemmatized_tokens'] = lemmatized
        
        # Extract entities
        if extract_entities:
            result['entities'] = self.extract_travel_entities(cleaned_text)
            result['pos_tags'] = self.extract_pos_tags(cleaned_text)
            result['named_entities'] = self.extract_named_entities(cleaned_text)
        
        # Analyze sentiment
        if analyze_sentiment:
            result['sentiment'] = self.analyze_sentiment(cleaned_text)
        
        return result
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[Dict]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (List[str]): List of texts to process
            **kwargs: Arguments for preprocessing pipeline
            
        Returns:
            List[Dict]: List of processed text data
        """
        results = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i}/{len(texts)}")
            results.append(self.preprocess_pipeline(text, **kwargs))
        return results

class TextPostprocessor:
    """
    Advanced text postprocessing for response generation.
    """
    
    def __init__(self):
        """Initialize the postprocessor."""
        self.response_templates = {
            'recommendation': "Based on your preferences, I recommend {destination}. {reason}",
            'information': "Here's what you need to know about {topic}: {information}",
            'comparison': "Comparing {item1} and {item2}: {comparison}",
            'general': "I'd be happy to help with your travel query. {response}"
        }
    
    def format_response(self, response: str, response_type: str = 'general', **kwargs) -> str:
        """
        Format response using templates.
        
        Args:
            response (str): Raw response
            response_type (str): Type of response
            **kwargs: Template variables
            
        Returns:
            str: Formatted response
        """
        if response_type in self.response_templates:
            template = self.response_templates[response_type]
            # Provide default values for missing keys
            safe_kwargs = {
                'destination': kwargs.get('destination', 'the destination'),
                'reason': kwargs.get('reason', 'It offers great experiences.'),
                'topic': kwargs.get('topic', 'this topic'),
                'information': kwargs.get('information', 'Here is the information you requested.'),
                'item1': kwargs.get('item1', 'item 1'),
                'item2': kwargs.get('item2', 'item 2'),
                'comparison': kwargs.get('comparison', 'Here is a comparison.'),
                'response': response
            }
            safe_kwargs.update(kwargs)  # Override with provided values
            return template.format(**safe_kwargs)
        return response
    
    def add_personalization(self, response: str, user_preferences: Dict) -> str:
        """
        Add personalization to response.
        
        Args:
            response (str): Response text
            user_preferences (Dict): User preferences
            
        Returns:
            str: Personalized response
        """
        if 'budget' in user_preferences:
            response += f"\n\n💡 Budget Tip: With a budget of ${user_preferences['budget']}, you can enjoy a comfortable trip."
        
        if 'interests' in user_preferences:
            interests = ', '.join(user_preferences['interests'])
            response += f"\n\n🎯 Based on your interests in {interests}, you might also enjoy..."
        
        return response
    
    def add_safety_considerations(self, response: str, destination: str) -> str:
        """
        Add safety considerations to response.
        
        Args:
            response (str): Response text
            destination (str): Destination name
            
        Returns:
            str: Response with safety information
        """
        safety_info = {
            'paris': "Paris is generally safe, but be aware of pickpockets in tourist areas.",
            'tokyo': "Tokyo is extremely safe with very low crime rates.",
            'london': "London is very safe, but keep valuables secure on public transport.",
            'new york': "New York is safe in tourist areas, but stay aware of your surroundings."
        }
        
        if destination.lower() in safety_info:
            response += f"\n\n⚠️ Safety: {safety_info[destination.lower()]}"
        
        return response

def main():
    """Demonstrate the advanced NLP preprocessing."""
    print("🧠 Advanced NLP Text Preprocessing Demo")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = AdvancedTextPreprocessor()
    postprocessor = TextPostprocessor()
    
    # Sample travel texts
    sample_texts = [
        "I want to visit Paris in March 2024. What's the best time to go?",
        "The hotel in Tokyo was amazing! 5 stars for sure. Cost me $200 per night.",
        "Compare London vs New York for a family vacation in December.",
        "Tell me about budget travel options in Europe under $100 per day."
    ]
    
    print("📝 Sample Texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    print()
    
    # Process each text
    for i, text in enumerate(sample_texts, 1):
        print(f"🔍 Processing Text {i}:")
        print(f"Original: {text}")
        
        # Preprocess
        processed = preprocessor.preprocess_pipeline(text)
        
        print(f"Cleaned: {processed['cleaned_text']}")
        print(f"Tokens: {processed['tokens'][:10]}...")  # Show first 10 tokens
        print(f"Entities: {processed['entities']}")
        print(f"Sentiment: {processed['sentiment']}")
        print()
    
    print("✅ Advanced NLP Preprocessing Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Text cleaning and normalization")
    print("  - Advanced tokenization (NLTK, spaCy)")
    print("  - Lemmatization and stemming")
    print("  - Stopword removal")
    print("  - Named entity recognition")
    print("  - Sentiment analysis")
    print("  - Travel-specific entity extraction")
    print("  - Part-of-speech tagging")

if __name__ == "__main__":
    main()
