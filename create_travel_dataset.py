#!/usr/bin/env python3
"""
Create comprehensive travel destinations dataset from all available data sources
"""

import pandas as pd
import numpy as np
import random
import os

def create_comprehensive_travel_dataset():
    print("🌍 Creating comprehensive travel destinations dataset...")
    
    # Load all available datasets
    print("📊 Loading datasets...")
    
    # 1. World cities data
    cities_df = pd.read_csv('data/world_cities.csv')
    print(f"✅ Loaded {len(cities_df)} world cities")
    
    # 2. Sri Lanka cities
    sri_lanka_df = pd.read_csv('data/sri_lanka_cities.csv')
    print(f"✅ Loaded {len(sri_lanka_df)} Sri Lanka cities")
    
    # 3. World GDP countries
    gdp_df = pd.read_csv('data/world_gdp_countries.csv')
    print(f"✅ Loaded {len(gdp_df)} countries with GDP data")
    
    # 4. USA states
    usa_states_df = pd.read_csv('data/usa_states.csv')
    print(f"✅ Loaded {len(usa_states_df)} USA states")
    
    # 5. Original travel datasets
    bitext_df = pd.read_csv('data/bitext-travel-llm-chatbot-training-dataset.csv')
    print(f"✅ Loaded {len(bitext_df)} travel Q&A pairs")
    
    tripadvisor_df = pd.read_csv('data/tripadvisor_review.csv')
    print(f"✅ Loaded {len(tripadvisor_df)} TripAdvisor reviews")
    
    # Create comprehensive travel destinations
    print("\n🏗️ Creating comprehensive travel destinations...")
    
    # Enhanced attraction templates with specific details
    attraction_templates = {
        'Cultural': [
            "Historic temples and ancient architecture dating back centuries",
            "Traditional markets with local crafts and authentic cuisine",
            "Cultural museums showcasing rich heritage and art",
            "Religious sites and spiritual centers of significance",
            "Local festivals and traditional celebrations throughout the year"
        ],
        'Natural': [
            "Pristine beaches with crystal-clear waters and golden sand",
            "Mountain ranges offering breathtaking hiking trails and views",
            "National parks with diverse wildlife and nature reserves",
            "Waterfalls and natural springs perfect for relaxation",
            "Scenic landscapes ideal for photography and outdoor activities"
        ],
        'Modern': [
            "Modern shopping districts with international brands",
            "Contemporary art galleries and cultural centers",
            "High-end restaurants serving fusion and international cuisine",
            "Entertainment venues and nightlife hotspots",
            "Business districts with modern architecture and amenities"
        ],
        'Adventure': [
            "Adventure sports facilities for hiking, diving, and water sports",
            "Wildlife safaris and nature exploration opportunities",
            "Mountain climbing and trekking routes for all skill levels",
            "Water activities including surfing, snorkeling, and fishing",
            "Cultural immersion experiences with local communities"
        ]
    }
    
    # Budget and time information
    budget_info = {
        'Budget': {'daily_cost': '$20-50', 'accommodation': 'Hostels, guesthouses', 'food': 'Street food, local restaurants'},
        'Mid-range': {'daily_cost': '$50-150', 'accommodation': '3-4 star hotels', 'food': 'Mix of local and international cuisine'},
        'Luxury': {'daily_cost': '$150-500+', 'accommodation': '5-star resorts, luxury hotels', 'food': 'Fine dining, gourmet experiences'}
    }
    
    best_time_info = {
        'Spring': 'March-May: Pleasant weather, blooming flowers, ideal for outdoor activities',
        'Summer': 'June-August: Warm weather, beach season, perfect for water activities',
        'Fall': 'September-November: Mild temperatures, harvest season, great for cultural tours',
        'Winter': 'December-February: Cool weather, holiday season, ideal for city exploration',
        'Year-round': 'Consistent climate throughout the year, suitable for all activities'
    }
    
    travel_data = []
    
    # Process world cities (focus on major destinations)
    print("🌎 Processing world cities...")
    major_cities = cities_df.head(1500)  # Focus on major cities
    
    for idx, row in major_cities.iterrows():
        destination = row['name']
        country = row['country']
        region = row['subcountry']
        
        # Determine category and attractions based on location
        if any(keyword in destination.lower() for keyword in ['beach', 'coast', 'bay', 'port']):
            category = 'Natural'
        elif any(keyword in destination.lower() for keyword in ['mountain', 'hill', 'peak', 'valley']):
            category = 'Adventure'
        elif any(keyword in destination.lower() for keyword in ['capital', 'city', 'town']):
            category = 'Cultural'
        else:
            category = random.choice(['Cultural', 'Natural', 'Modern', 'Adventure'])
        
        # Select appropriate attractions
        attractions = random.choice(attraction_templates[category])
        
        # Determine budget level
        budget = random.choice(['Budget', 'Mid-range', 'Luxury'])
        best_time = random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Year-round'])
        
        # Generate realistic ratings and reviews
        rating = round(np.random.uniform(3.8, 4.8), 1)
        reviews = random.randint(100, 5000)
        
        # Create comprehensive description
        description = f"{destination} in {country} is a {category.lower()} destination offering {attractions.lower()}. "
        description += f"Perfect for {budget.lower()} travelers with {budget_info[budget]['daily_cost']} daily budget. "
        description += f"{best_time_info[best_time]}"
        
        travel_data.append({
            'destination': destination,
            'country': country,
            'region': region,
            'category': category,
            'attractions': attractions,
            'best_time': best_time,
            'best_time_description': best_time_info[best_time],
            'budget': budget,
            'daily_cost': budget_info[budget]['daily_cost'],
            'accommodation': budget_info[budget]['accommodation'],
            'food': budget_info[budget]['food'],
            'description': description,
            'rating': rating,
            'reviews': reviews,
            'type': 'destination'
        })
    
    # Process Sri Lanka cities from our dataset
    print("🇱🇰 Processing Sri Lanka cities from dataset...")
    for idx, row in sri_lanka_df.iterrows():
        destination = row['name']
        country = 'Sri Lanka'
        region = row.get('subcountry', 'Unknown')
        
        # Determine category based on city name
        if any(keyword in destination.lower() for keyword in ['beach', 'coast', 'bay', 'port']):
            category = 'Natural'
        elif any(keyword in destination.lower() for keyword in ['mountain', 'hill', 'peak', 'valley']):
            category = 'Adventure'
        else:
            category = 'Cultural'
        
        # Select appropriate attractions
        attractions = random.choice(attraction_templates[category])
        
        # Determine budget level
        budget = random.choice(['Budget', 'Mid-range', 'Luxury'])
        best_time = random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Year-round'])
        
        # Generate realistic ratings and reviews
        rating = round(np.random.uniform(3.8, 4.8), 1)
        reviews = random.randint(100, 3000)
        
        # Create comprehensive description
        description = f"{destination} in {country} is a {category.lower()} destination offering {attractions.lower()}. "
        description += f"Perfect for {budget.lower()} travelers with {budget_info[budget]['daily_cost']} daily budget. "
        description += f"{best_time_info[best_time]}"
        
        travel_data.append({
            'destination': destination,
            'country': country,
            'region': region,
            'category': category,
            'attractions': attractions,
            'best_time': best_time,
            'best_time_description': best_time_info[best_time],
            'budget': budget,
            'daily_cost': budget_info[budget]['daily_cost'],
            'accommodation': budget_info[budget]['accommodation'],
            'food': budget_info[budget]['food'],
            'description': description,
            'rating': rating,
            'reviews': reviews,
            'type': 'destination'
        })
    
    # Process original travel datasets for additional destinations
    print("📚 Processing original travel datasets...")
    
    # Extract destinations from bitext dataset
    if len(bitext_df) > 0:
        print(f"Processing {len(bitext_df)} travel Q&A pairs...")
        # Sample some destinations from the bitext dataset
        sample_size = min(500, len(bitext_df))
        bitext_sample = bitext_df.sample(n=sample_size, random_state=42)
        
        for idx, row in bitext_sample.iterrows():
            # Extract destination information from questions/answers
            question = str(row.get('question', ''))
            answer = str(row.get('answer', ''))
            
            # Simple extraction of destination names (basic approach)
            destinations = []
            for city_row in cities_df.head(100).iterrows():
                city_name = city_row[1]['name']
                if city_name.lower() in question.lower() or city_name.lower() in answer.lower():
                    destinations.append(city_name)
            
            if destinations:
                destination = destinations[0]  # Take first match
                country = cities_df[cities_df['name'] == destination]['country'].iloc[0] if len(cities_df[cities_df['name'] == destination]) > 0 else 'Unknown'
                region = cities_df[cities_df['name'] == destination]['subcountry'].iloc[0] if len(cities_df[cities_df['name'] == destination]) > 0 else 'Unknown'
                
                # Create travel entry based on real Q&A data
                category = 'Cultural'  # Default
                attractions = f"Travel destination mentioned in travel Q&A: {answer[:100]}..."
                budget = random.choice(['Budget', 'Mid-range', 'Luxury'])
                best_time = random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Year-round'])
                rating = round(np.random.uniform(3.5, 4.9), 1)
                reviews = random.randint(50, 2000)
                
                description = f"{destination} is a travel destination. {answer[:200]}..."
                
                travel_data.append({
                    'destination': destination,
                    'country': country,
                    'region': region,
                    'category': category,
                    'attractions': attractions,
                    'best_time': best_time,
                    'best_time_description': best_time_info[best_time],
                    'budget': budget,
                    'daily_cost': budget_info[budget]['daily_cost'],
                    'accommodation': budget_info[budget]['accommodation'],
                    'food': budget_info[budget]['food'],
                    'description': description,
                    'rating': rating,
                    'reviews': reviews,
                    'type': 'destination'
                })
    
    # Process TripAdvisor reviews for additional destinations
    if len(tripadvisor_df) > 0:
        print(f"Processing {len(tripadvisor_df)} TripAdvisor reviews...")
        # Sample some reviews
        sample_size = min(200, len(tripadvisor_df))
        tripadvisor_sample = tripadvisor_df.sample(n=sample_size, random_state=42)
        
        for idx, row in tripadvisor_sample.iterrows():
            # Extract destination from review data
            review_text = str(row.get('Review', ''))
            rating_val = row.get('Rating', 4.0)
            
            # Find matching cities in our dataset
            destinations = []
            for city_row in cities_df.head(50).iterrows():
                city_name = city_row[1]['name']
                if city_name.lower() in review_text.lower():
                    destinations.append(city_name)
            
            if destinations:
                destination = destinations[0]
                country = cities_df[cities_df['name'] == destination]['country'].iloc[0] if len(cities_df[cities_df['name'] == destination]) > 0 else 'Unknown'
                region = cities_df[cities_df['name'] == destination]['subcountry'].iloc[0] if len(cities_df[cities_df['name'] == destination]) > 0 else 'Unknown'
                
                # Create travel entry based on real review data
                category = 'Cultural'
                attractions = f"Destination with real traveler reviews: {review_text[:100]}..."
                budget = random.choice(['Budget', 'Mid-range', 'Luxury'])
                best_time = random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Year-round'])
                rating = float(rating_val) if rating_val else round(np.random.uniform(3.5, 4.9), 1)
                reviews = random.randint(100, 1500)
                
                description = f"{destination} based on real traveler reviews. {review_text[:200]}..."
                
                travel_data.append({
                    'destination': destination,
                    'country': country,
                    'region': region,
                    'category': category,
                    'attractions': attractions,
                    'best_time': best_time,
                    'best_time_description': best_time_info[best_time],
                    'budget': budget,
                    'daily_cost': budget_info[budget]['daily_cost'],
                    'accommodation': budget_info[budget]['accommodation'],
                    'food': budget_info[budget]['food'],
                    'description': description,
                    'rating': rating,
                    'reviews': reviews,
                    'type': 'destination'
                })
    
    # Create DataFrame
    travel_df = pd.DataFrame(travel_data)
    
    # Save comprehensive dataset
    travel_df.to_csv('data/comprehensive_travel_destinations.csv', index=False)
    print(f"✅ Created comprehensive_travel_destinations.csv with {len(travel_df)} destinations")
    
    # Create a separate Sri Lanka focused dataset
    sri_lanka_focused = travel_df[travel_df['country'] == 'Sri Lanka']
    sri_lanka_focused.to_csv('data/sri_lanka_travel_guide.csv', index=False)
    print(f"✅ Created sri_lanka_travel_guide.csv with {len(sri_lanka_focused)} Sri Lanka destinations")
    
    # Show sample data
    print("\n📋 Sample data:")
    print(travel_df.head(3))
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total destinations: {len(travel_df)}")
    print(f"Countries covered: {travel_df['country'].nunique()}")
    print(f"Average rating: {travel_df['rating'].mean():.2f}")
    print(f"Budget distribution:")
    print(travel_df['budget'].value_counts())
    print(f"Category distribution:")
    print(travel_df['category'].value_counts())
    print(f"Sri Lanka destinations: {len(sri_lanka_focused)}")
    
    return travel_df

if __name__ == "__main__":
    create_comprehensive_travel_dataset()
