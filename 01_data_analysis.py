"""
Step 2: Data Analysis - Travel Advisor Chatbot
Analyze the real datasets to understand their structure and content
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

def analyze_bitext_dataset():
    """Analyze the Bitext travel Q&A dataset."""
    print("🔍 Analyzing Bitext Travel Q&A Dataset...")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('data/bitext-travel-llm-chatbot-training-dataset.csv')
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📊 Total Records: {len(df):,}")
    print(f"📊 Columns: {list(df.columns)}")
    print()
    
    # Basic info
    print("📋 Column Information:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
        print(f"    Non-null: {df[col].notna().sum():,} / {len(df):,}")
        print(f"    Unique values: {df[col].nunique():,}")
        print()
    
    # Sample data
    print("📝 Sample Data:")
    print(df.head(3).to_string())
    print()
    
    # Analyze instruction column
    if 'instruction' in df.columns:
        print("📝 Instruction Analysis:")
        instruction_lengths = df['instruction'].str.len()
        print(f"  - Average length: {instruction_lengths.mean():.1f} characters")
        print(f"  - Min length: {instruction_lengths.min()} characters")
        print(f"  - Max length: {instruction_lengths.max()} characters")
        print()
        
        # Sample instructions
        print("📝 Sample Instructions:")
        for i, instruction in enumerate(df['instruction'].head(5)):
            print(f"  {i+1}. {instruction[:100]}...")
        print()
    
    # Analyze response column
    if 'response' in df.columns:
        print("📝 Response Analysis:")
        response_lengths = df['response'].str.len()
        print(f"  - Average length: {response_lengths.mean():.1f} characters")
        print(f"  - Min length: {response_lengths.min()} characters")
        print(f"  - Max length: {response_lengths.max()} characters")
        print()
        
        # Sample responses
        print("📝 Sample Responses:")
        for i, response in enumerate(df['response'].head(3)):
            print(f"  {i+1}. {response[:150]}...")
        print()
    
    # Analyze intent column
    if 'intent' in df.columns:
        print("🎯 Intent Analysis:")
        intent_counts = df['intent'].value_counts()
        print(f"  - Total unique intents: {len(intent_counts)}")
        print("  - Top 10 intents:")
        for intent, count in intent_counts.head(10).items():
            print(f"    {intent}: {count:,} ({count/len(df)*100:.1f}%)")
        print()
    
    # Analyze category column
    if 'category' in df.columns:
        print("📂 Category Analysis:")
        category_counts = df['category'].value_counts()
        print(f"  - Total unique categories: {len(category_counts)}")
        print("  - Top 10 categories:")
        for category, count in category_counts.head(10).items():
            print(f"    {category}: {count:,} ({count/len(df)*100:.1f}%)")
        print()
    
    return df

def analyze_tripadvisor_dataset():
    """Analyze the TripAdvisor reviews dataset."""
    print("🔍 Analyzing TripAdvisor Reviews Dataset...")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('data/tripadvisor_review.csv')
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📊 Total Records: {len(df):,}")
    print(f"📊 Columns: {list(df.columns)}")
    print()
    
    # Basic info
    print("📋 Column Information:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
        print(f"    Non-null: {df[col].notna().sum():,} / {len(df):,}")
        print(f"    Unique values: {df[col].nunique():,}")
        print()
    
    # Sample data
    print("📝 Sample Data:")
    print(df.head(3).to_string())
    print()
    
    # Analyze the data structure
    print("📊 Data Structure Analysis:")
    print(f"  - This appears to be a user rating matrix")
    print(f"  - Columns represent different categories/features")
    print(f"  - Values appear to be ratings (0-5 scale)")
    print()
    
    # Analyze rating patterns
    print("⭐ Rating Analysis:")
    for col in df.columns[1:]:  # Skip User ID column
        if df[col].dtype in ['float64', 'int64']:
            print(f"  - {col}:")
            print(f"    Mean: {df[col].mean():.2f}")
            print(f"    Min: {df[col].min():.2f}")
            print(f"    Max: {df[col].max():.2f}")
            print(f"    Non-zero values: {(df[col] > 0).sum():,}")
    print()
    
    return df

def create_sample_travel_data():
    """Create sample travel data for demonstration."""
    print("📝 Creating Sample Travel Data for Demonstration...")
    print("=" * 50)
    
    # Sample travel destinations with reviews
    sample_data = {
        'location': [
            'Paris', 'Tokyo', 'London', 'New York', 'Sydney',
            'Rome', 'Barcelona', 'Amsterdam', 'Bangkok', 'Dubai',
            'Singapore', 'Istanbul', 'Prague', 'Vienna', 'Berlin'
        ],
        'country': [
            'France', 'Japan', 'UK', 'USA', 'Australia',
            'Italy', 'Spain', 'Netherlands', 'Thailand', 'UAE',
            'Singapore', 'Turkey', 'Czech Republic', 'Austria', 'Germany'
        ],
        'review_text': [
            'Amazing city with beautiful architecture, great food, and rich culture. The Eiffel Tower and Louvre are must-sees.',
            'Incredible blend of traditional and modern. Great food, efficient transportation, and friendly people.',
            'Historic city with world-class museums, great theater, and diverse neighborhoods. Weather can be unpredictable.',
            'The city that never sleeps! Amazing food scene, Broadway shows, and endless activities.',
            'Beautiful harbor city with great beaches, outdoor lifestyle, and iconic landmarks like the Opera House.',
            'Ancient history comes alive here. Amazing food, beautiful architecture, and rich cultural heritage.',
            'Vibrant city with amazing architecture, great food scene, and beautiful beaches. Perfect for art lovers.',
            'Charming canals, beautiful architecture, and great museums. Perfect for a romantic getaway.',
            'Incredible street food, beautiful temples, and friendly people. Great value for money.',
            'Modern city with amazing architecture, luxury shopping, and great desert experiences.',
            'Clean, modern city with amazing food, great shopping, and efficient public transport.',
            'Where East meets West. Amazing history, beautiful mosques, and incredible food.',
            'Fairytale city with beautiful architecture, great beer, and charming old town.',
            'Elegant city with amazing music, beautiful palaces, and great coffee culture.',
            'Vibrant city with amazing nightlife, great museums, and rich history.'
        ],
        'rating': [5, 5, 4, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5, 4, 4],
        'category': [
            'Cultural', 'Cultural', 'Cultural', 'Entertainment', 'Nature',
            'Cultural', 'Cultural', 'Romantic', 'Budget', 'Luxury',
            'Modern', 'Cultural', 'Cultural', 'Cultural', 'Nightlife'
        ],
        'budget_level': [
            'Mid-range', 'Mid-range', 'High', 'High', 'Mid-range',
            'Mid-range', 'Mid-range', 'Mid-range', 'Budget', 'Luxury',
            'High', 'Mid-range', 'Budget', 'Mid-range', 'Mid-range'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    print(f"📊 Created sample dataset with {len(sample_df)} destinations")
    print("📝 Sample data:")
    print(sample_df.head(3).to_string())
    print()
    
    # Save sample data
    sample_df.to_csv('data/sample_travel_destinations.csv', index=False)
    print("✅ Sample data saved to: data/sample_travel_destinations.csv")
    
    return sample_df

def main():
    """Main analysis function."""
    print("🌍 Travel Advisor Chatbot - Data Analysis")
    print("=" * 60)
    print()
    
    # Analyze Bitext dataset
    bitext_df = analyze_bitext_dataset()
    print()
    
    # Analyze TripAdvisor dataset
    tripadvisor_df = analyze_tripadvisor_dataset()
    print()
    
    # Create sample travel data
    sample_df = create_sample_travel_data()
    print()
    
    # Summary
    print("📊 Analysis Summary:")
    print("=" * 30)
    print(f"✅ Bitext Dataset: {len(bitext_df):,} Q&A pairs")
    print(f"✅ TripAdvisor Dataset: {len(tripadvisor_df):,} user ratings")
    print(f"✅ Sample Travel Data: {len(sample_df)} destinations")
    print()
    print("🎯 Key Insights:")
    print("  - Bitext dataset contains rich Q&A pairs for training")
    print("  - TripAdvisor dataset provides user rating patterns")
    print("  - Sample data provides destination information")
    print("  - Ready for preprocessing and model training!")
    print()
    print("✅ Step 2 Complete: Data Analysis Done!")

if __name__ == "__main__":
    main()
