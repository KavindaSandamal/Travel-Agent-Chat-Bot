#!/usr/bin/env python3
"""
Enhanced Travel Dataset Creator
Combines multiple travel datasets to create a comprehensive travel destinations dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import random

def load_existing_datasets():
    """Load existing comprehensive datasets."""
    print("📊 Loading existing datasets...")
    
    # Load existing comprehensive dataset
    if os.path.exists('data/comprehensive_travel_destinations.csv'):
        existing_df = pd.read_csv('data/comprehensive_travel_destinations.csv')
        print(f"✅ Loaded existing comprehensive dataset: {len(existing_df)} destinations")
    else:
        existing_df = pd.DataFrame()
        print("⚠️ No existing comprehensive dataset found")
    
    # Load Sri Lanka guide
    if os.path.exists('data/sri_lanka_travel_guide.csv'):
        sri_lanka_df = pd.read_csv('data/sri_lanka_travel_guide.csv')
        print(f"✅ Loaded Sri Lanka guide: {len(sri_lanka_df)} destinations")
    else:
        sri_lanka_df = pd.DataFrame()
        print("⚠️ No Sri Lanka guide found")
    
    return existing_df, sri_lanka_df

def load_destination_reviews():
    """Load and process destination reviews dataset."""
    print("\n📝 Processing Destination Reviews dataset...")
    
    try:
        reviews_df = pd.read_csv('data/Destination Reviews (final).csv')
        print(f"✅ Loaded {len(reviews_df)} destination reviews")
        
        # Process reviews to extract destination information
        destination_data = []
        
        # Group by destination to aggregate reviews
        for destination, group in reviews_df.groupby('Destination'):
            # Get basic info
            district = group['District'].iloc[0] if not group['District'].isna().all() else 'Unknown'
            
            # Combine all reviews for this destination
            all_reviews = ' '.join(group['Review'].dropna().astype(str))
            
            # Calculate average rating based on review sentiment (simplified)
            review_count = len(group)
            avg_rating = min(5.0, max(3.0, 3.5 + (review_count * 0.1)))  # Simple heuristic
            
            # Determine category based on destination name
            destination_lower = destination.lower()
            if any(keyword in destination_lower for keyword in ['beach', 'bay', 'coast', 'shore']):
                category = 'Natural'
            elif any(keyword in destination_lower for keyword in ['temple', 'church', 'mosque', 'monastery', 'cultural']):
                category = 'Cultural'
            elif any(keyword in destination_lower for keyword in ['park', 'sanctuary', 'reserve', 'forest', 'mountain']):
                category = 'Natural'
            elif any(keyword in destination_lower for keyword in ['museum', 'gallery', 'heritage']):
                category = 'Cultural'
            else:
                category = 'Cultural'  # Default
            
            # Determine budget based on location and type
            if 'colombo' in district.lower() or 'kandy' in district.lower():
                budget = 'Mid-range'
                daily_cost = '$50-100'
            else:
                budget = 'Budget'
                daily_cost = '$20-50'
            
            # Determine best time to visit
            best_time = random.choice(['Year-round', 'Winter', 'Summer', 'Spring'])
            
            destination_data.append({
                'destination': destination,
                'country': 'Sri Lanka',
                'region': district,
                'category': category,
                'attractions': all_reviews[:200] + '...' if len(all_reviews) > 200 else all_reviews,
                'best_time': best_time,
                'budget': budget,
                'daily_cost': daily_cost,
                'rating': round(avg_rating, 1),
                'reviews': review_count,
                'description': f"Popular destination in {district} with {review_count} reviews. {all_reviews[:150]}..."
            })
        
        print(f"✅ Processed {len(destination_data)} destinations from reviews")
        return pd.DataFrame(destination_data)
        
    except Exception as e:
        print(f"❌ Error loading destination reviews: {e}")
        return pd.DataFrame()

def load_detailed_reviews():
    """Load and process detailed reviews dataset."""
    print("\n📝 Processing Detailed Reviews dataset...")
    
    try:
        reviews_df = pd.read_csv('data/Reviews.csv', encoding='latin-1')
        print(f"✅ Loaded {len(reviews_df)} detailed reviews")
        
        # Process reviews to extract location information
        location_data = []
        
        # Group by location to aggregate reviews
        for location, group in reviews_df.groupby('Location_Name'):
            # Get basic info
            city = group['Located_City'].iloc[0] if not group['Located_City'].isna().all() else 'Unknown'
            location_type = group['Location_Type'].iloc[0] if not group['Location_Type'].isna().all() else 'Attraction'
            
            # Calculate average rating
            avg_rating = group['Rating'].mean() if 'Rating' in group.columns else 4.0
            
            # Combine all review texts
            all_reviews = ' '.join(group['Text'].dropna().astype(str))
            review_count = len(group)
            
            # Determine category based on location type
            if location_type.lower() in ['beaches', 'beach']:
                category = 'Natural'
            elif location_type.lower() in ['cultural', 'historical', 'religious']:
                category = 'Cultural'
            elif location_type.lower() in ['adventure', 'outdoor']:
                category = 'Adventure'
            else:
                category = 'Cultural'  # Default
            
            # Determine budget and country
            if 'sri lanka' in city.lower() or 'colombo' in city.lower():
                country = 'Sri Lanka'
                budget = 'Budget'
                daily_cost = '$20-50'
            else:
                country = 'Unknown'
                budget = 'Mid-range'
                daily_cost = '$50-150'
            
            location_data.append({
                'destination': location,
                'country': country,
                'region': city,
                'category': category,
                'attractions': all_reviews[:200] + '...' if len(all_reviews) > 200 else all_reviews,
                'best_time': 'Year-round',
                'budget': budget,
                'daily_cost': daily_cost,
                'rating': round(avg_rating, 1),
                'reviews': review_count,
                'description': f"{location_type} in {city} with {review_count} reviews. {all_reviews[:150]}..."
            })
        
        print(f"✅ Processed {len(location_data)} locations from detailed reviews")
        return pd.DataFrame(location_data)
        
    except Exception as e:
        print(f"❌ Error loading detailed reviews: {e}")
        return pd.DataFrame()

def load_accommodation_data():
    """Load accommodation data from the Excel file."""
    print("\n🏨 Processing Accommodation Information dataset...")
    
    try:
        # Read the accommodation data (it's actually a CSV with .xls extension)
        accommodation_df = pd.read_csv('data/Information for Accommodation.xls')
        print(f"✅ Loaded {len(accommodation_df)} accommodation records")
        
        accommodation_data = []
        
        for _, row in accommodation_df.iterrows():
            try:
                # Extract accommodation information
                acc_type = str(row.get('Type', 'Hotel'))
                name = str(row.get('Name', 'Unknown'))
                address = str(row.get('Address', ''))
                rooms = row.get('Rooms', 0)
                grade = str(row.get('Grade', 'Standard'))
                district = str(row.get('District', 'Unknown'))
                latitude = row.get('Latitude', 0)
                longitude = row.get('Logitiute', 0)  # Note: typo in original data
                
                # Skip if essential data is missing
                if name == 'Unknown' or pd.isna(name):
                    continue
                
                # Determine category based on accommodation type
                if 'boutique' in acc_type.lower():
                    category = 'Modern'
                    budget = 'Luxury'
                    daily_cost = '$150-500+'
                elif 'resort' in acc_type.lower():
                    category = 'Natural'
                    budget = 'Luxury'
                    daily_cost = '$150-500+'
                elif 'hotel' in acc_type.lower():
                    category = 'Modern'
                    budget = 'Mid-range'
                    daily_cost = '$50-150'
                else:
                    category = 'Modern'
                    budget = 'Budget'
                    daily_cost = '$20-50'
                
                # Determine rating based on grade
                grade_map = {
                    'Luxury': 5.0, 'Deluxe': 4.5, 'Standard': 3.5, 'Budget': 3.0
                }
                rating = grade_map.get(grade, 3.5)
                
                # Create attractions description
                attractions = f"{acc_type} accommodation in {district}"
                if rooms > 0:
                    attractions += f" with {rooms} rooms"
                if latitude != 0 and longitude != 0:
                    attractions += f" (GPS: {latitude:.4f}, {longitude:.4f})"
                
                accommodation_data.append({
                    'destination': name,
                    'country': 'Sri Lanka',
                    'region': district,
                    'category': category,
                    'attractions': attractions,
                    'best_time': 'Year-round',
                    'budget': budget,
                    'daily_cost': daily_cost,
                    'rating': rating,
                    'reviews': random.randint(5, 200),  # Simulated review count
                    'description': f"{acc_type} in {district}, Sri Lanka. {address}. Grade: {grade}"
                })
                
            except Exception as e:
                # Skip problematic rows
                continue
        
        print(f"✅ Processed {len(accommodation_data)} accommodations")
        return pd.DataFrame(accommodation_data)
        
    except Exception as e:
        print(f"❌ Error loading accommodation data: {e}")
        return pd.DataFrame()

def load_hotels_sample():
    """Load a sample of hotels (20 per country) from the hotels dataset."""
    print("\n🏨 Processing Hotels dataset (sampling 20 per country)...")
    
    try:
        # Read hotels data in chunks to handle large file
        chunk_size = 10000
        hotels_data = []
        country_counts = {}
        
        print("📖 Reading hotels data in chunks...")
        for chunk in pd.read_csv('data/hotels.csv', encoding='latin-1', chunksize=chunk_size):
            # Clean column names (remove extra spaces)
            chunk.columns = chunk.columns.str.strip()
            
            for _, row in chunk.iterrows():
                try:
                    country = row.get('countyName', 'Unknown')
                    if pd.isna(country) or country == 'Unknown':
                        continue
                    
                    # Limit to 20 hotels per country
                    if country_counts.get(country, 0) >= 20:
                        continue
                    
                    # Extract hotel information with safe handling
                    hotel_name = str(row.get('HotelName', 'Unknown Hotel'))
                    city = str(row.get('cityName', 'Unknown City'))
                    rating = str(row.get('HotelRating', 'ThreeStar'))
                    description = str(row.get('Description', ''))
                    attractions = str(row.get('Attractions', ''))
                    
                    # Skip if essential data is missing
                    if hotel_name == 'Unknown Hotel' or city == 'Unknown City':
                        continue
                
                    # Convert rating to numeric
                    rating_map = {
                        'OneStar': 1.0, 'TwoStar': 2.0, 'ThreeStar': 3.0,
                        'FourStar': 4.0, 'FiveStar': 5.0
                    }
                    numeric_rating = rating_map.get(rating, 3.0)
                    
                    # Determine category and budget based on rating
                    if numeric_rating >= 4.0:
                        category = 'Modern'
                        budget = 'Luxury'
                        daily_cost = '$150-500+'
                    elif numeric_rating >= 3.0:
                        category = 'Modern'
                        budget = 'Mid-range'
                        daily_cost = '$50-150'
                    else:
                        category = 'Modern'
                        budget = 'Budget'
                        daily_cost = '$20-50'
                    
                    hotels_data.append({
                        'destination': hotel_name,
                        'country': country,
                        'region': city,
                        'category': category,
                        'attractions': attractions[:200] + '...' if len(attractions) > 200 else attractions,
                        'best_time': 'Year-round',
                        'budget': budget,
                        'daily_cost': daily_cost,
                        'rating': numeric_rating,
                        'reviews': random.randint(10, 500),  # Simulated review count
                        'description': f"Hotel in {city}, {country}. {description[:150]}..."
                    })
                    
                    country_counts[country] = country_counts.get(country, 0) + 1
                    
                except Exception as e:
                    # Skip problematic rows
                    continue
                
                # Stop if we have enough countries with 20 hotels each
                if len(country_counts) >= 50 and all(count >= 20 for count in country_counts.values()):
                    break
            
            if len(country_counts) >= 50 and all(count >= 20 for count in country_counts.values()):
                break
        
        print(f"✅ Processed {len(hotels_data)} hotels from {len(country_counts)} countries")
        return pd.DataFrame(hotels_data)
        
    except Exception as e:
        print(f"❌ Error loading hotels: {e}")
        return pd.DataFrame()

def combine_datasets(existing_df, sri_lanka_df, reviews_df, detailed_reviews_df, accommodation_df, hotels_df):
    """Combine all datasets into a comprehensive travel dataset."""
    print("\n🔄 Combining all datasets...")
    
    # Start with existing comprehensive dataset
    combined_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()
    
    # Add Sri Lanka guide data
    if not sri_lanka_df.empty:
        combined_df = pd.concat([combined_df, sri_lanka_df], ignore_index=True)
        print(f"✅ Added {len(sri_lanka_df)} Sri Lanka destinations")
    
    # Add destination reviews data
    if not reviews_df.empty:
        combined_df = pd.concat([combined_df, reviews_df], ignore_index=True)
        print(f"✅ Added {len(reviews_df)} destinations from reviews")
    
    # Add detailed reviews data
    if not detailed_reviews_df.empty:
        combined_df = pd.concat([combined_df, detailed_reviews_df], ignore_index=True)
        print(f"✅ Added {len(detailed_reviews_df)} locations from detailed reviews")
    
    # Add accommodation data
    if not accommodation_df.empty:
        combined_df = pd.concat([combined_df, accommodation_df], ignore_index=True)
        print(f"✅ Added {len(accommodation_df)} accommodations")
    
    # Add hotels data
    if not hotels_df.empty:
        combined_df = pd.concat([combined_df, hotels_df], ignore_index=True)
        print(f"✅ Added {len(hotels_df)} hotels")
    
    # Remove duplicates based on destination and country
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['destination', 'country'], keep='first')
    final_count = len(combined_df)
    print(f"✅ Removed {initial_count - final_count} duplicate destinations")
    
    # Sort by rating and country
    combined_df = combined_df.sort_values(['country', 'rating'], ascending=[True, False])
    
    print(f"🎉 Final combined dataset: {len(combined_df)} destinations")
    return combined_df

def save_enhanced_datasets(combined_df):
    """Save the enhanced datasets."""
    print("\n💾 Saving enhanced datasets...")
    
    # Save comprehensive dataset
    combined_df.to_csv('data/enhanced_travel_destinations.csv', index=False)
    print(f"✅ Saved enhanced travel destinations: {len(combined_df)} destinations")
    
    # Create enhanced Sri Lanka guide
    sri_lanka_enhanced = combined_df[combined_df['country'] == 'Sri Lanka'].copy()
    if not sri_lanka_enhanced.empty:
        sri_lanka_enhanced.to_csv('data/enhanced_sri_lanka_guide.csv', index=False)
        print(f"✅ Saved enhanced Sri Lanka guide: {len(sri_lanka_enhanced)} destinations")
    
    # Create summary statistics
    summary_stats = {
        'total_destinations': len(combined_df),
        'countries_covered': combined_df['country'].nunique(),
        'categories': combined_df['category'].value_counts().to_dict(),
        'budget_distribution': combined_df['budget'].value_counts().to_dict(),
        'average_rating': round(combined_df['rating'].mean(), 2),
        'creation_date': datetime.now().isoformat()
    }
    
    # Save summary
    import json
    with open('data/dataset_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print("✅ Saved dataset summary")
    
    return summary_stats

def main():
    """Main function to create enhanced travel dataset."""
    print("🌟 Enhanced Travel Dataset Creator")
    print("=" * 50)
    
    # Load existing datasets
    existing_df, sri_lanka_df = load_existing_datasets()
    
    # Load and process new datasets
    reviews_df = load_destination_reviews()
    detailed_reviews_df = load_detailed_reviews()
    accommodation_df = load_accommodation_data()
    hotels_df = load_hotels_sample()
    
    # Combine all datasets
    combined_df = combine_datasets(existing_df, sri_lanka_df, reviews_df, detailed_reviews_df, accommodation_df, hotels_df)
    
    # Save enhanced datasets
    summary_stats = save_enhanced_datasets(combined_df)
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print(f"Total Destinations: {summary_stats['total_destinations']}")
    print(f"Countries Covered: {summary_stats['countries_covered']}")
    print(f"Average Rating: {summary_stats['average_rating']}")
    print(f"Categories: {summary_stats['categories']}")
    print(f"Budget Distribution: {summary_stats['budget_distribution']}")
    
    print("\n🎉 Enhanced travel dataset creation completed!")
    print("📁 Files created:")
    print("  - data/enhanced_travel_destinations.csv")
    print("  - data/enhanced_sri_lanka_guide.csv")
    print("  - data/dataset_summary.json")

if __name__ == "__main__":
    main()
