"""
Advanced Prompt Engineering for Travel Advisor Chatbot
Demonstrates sophisticated prompt techniques: templates, few-shot, chain-of-thought, dynamic generation
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import random
import string
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PromptType(Enum):
    """Types of prompts for different travel scenarios."""
    RECOMMENDATION = "recommendation"
    INFORMATION = "information"
    COMPARISON = "comparison"
    PLANNING = "planning"
    BOOKING = "booking"
    WEATHER = "weather"
    SAFETY = "safety"
    BUDGET = "budget"
    ACTIVITIES = "activities"
    GENERAL = "general"

@dataclass
class PromptTemplate:
    """Data class for prompt templates."""
    name: str
    template: str
    variables: List[str]
    description: str
    prompt_type: PromptType
    examples: List[Dict[str, str]] = None

class TravelPromptEngineer:
    """
    Advanced prompt engineering system for travel recommendations.
    """
    
    def __init__(self):
        """Initialize the prompt engineer."""
        self.templates = self._initialize_templates()
        self.few_shot_examples = self._initialize_few_shot_examples()
        self.chain_of_thought_examples = self._initialize_chain_of_thought_examples()
        self.dynamic_prompts = {}
        self.user_profiles = {}
        
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize comprehensive prompt templates."""
        templates = {}
        
        # Recommendation Template
        templates['recommendation'] = PromptTemplate(
            name="travel_recommendation",
            template="""You are an expert travel advisor with deep knowledge of destinations worldwide. Based on the following context and user preferences, provide personalized travel recommendations.

Context Information:
{context}

User Query: {query}
User Preferences: {preferences}
Destination Focus: {destination}
Budget Range: {budget}
Travel Style: {travel_style}
Group Size: {group_size}

Please provide:
1. **Top 3 Destination Recommendations** with specific reasons
2. **Best Time to Visit** for each destination
3. **Estimated Budget** breakdown (accommodation, food, activities)
4. **Must-See Attractions** for each destination
5. **Local Tips** and hidden gems
6. **Safety Considerations** if applicable
7. **Cultural Insights** and etiquette tips

Format your response in a clear, engaging manner that helps the user make informed travel decisions.""",
            variables=['context', 'query', 'preferences', 'destination', 'budget', 'travel_style', 'group_size'],
            description="Template for generating travel recommendations",
            prompt_type=PromptType.RECOMMENDATION
        )
        
        # Information Template
        templates['information'] = PromptTemplate(
            name="travel_information",
            template="""You are a knowledgeable travel expert. Provide comprehensive information about the requested destination or topic.

Context Information:
{context}

User Query: {query}
Destination: {destination}
Information Type: {info_type}
User Level: {user_level}

Please provide detailed information about:
1. **Overview** of the destination/topic
2. **Key Highlights** and unique features
3. **Practical Information** (visa requirements, currency, language)
4. **Best Time to Visit** and weather considerations
5. **Getting Around** transportation options
6. **Cultural Insights** and local customs
7. **Safety and Health** considerations
8. **Local Laws** and important regulations

Ensure your response is accurate, helpful, and engaging.""",
            variables=['context', 'query', 'destination', 'info_type', 'user_level'],
            description="Template for providing travel information",
            prompt_type=PromptType.INFORMATION
        )
        
        # Comparison Template
        templates['comparison'] = PromptTemplate(
            name="travel_comparison",
            template="""You are an expert travel advisor specializing in destination comparisons. Analyze the requested destinations and provide a detailed comparison.

Context Information:
{context}

User Query: {query}
Destinations to Compare: {destinations}
Comparison Criteria: {criteria}
User Preferences: {preferences}

Please provide a comprehensive comparison covering:
1. **Overview** of each destination
2. **Cost Comparison** (budget, mid-range, luxury options)
3. **Best Time to Visit** for each
4. **Key Attractions** and activities
5. **Pros and Cons** of each destination
6. **Recommendation** based on user preferences
7. **Alternative Options** if applicable
8. **Decision Matrix** with scoring

Use a clear, structured format that makes it easy to compare options.""",
            variables=['context', 'query', 'destinations', 'criteria', 'preferences'],
            description="Template for comparing travel destinations",
            prompt_type=PromptType.COMPARISON
        )
        
        # Planning Template
        templates['planning'] = PromptTemplate(
            name="travel_planning",
            template="""You are a professional travel planner. Create a detailed itinerary based on the user's requirements.

Context Information:
{context}

User Query: {query}
Trip Details: {trip_details}
Duration: {duration}
Budget: {budget}
Interests: {interests}
Group Composition: {group_composition}

Please create a comprehensive travel plan including:
1. **Day-by-Day Itinerary** with time allocations
2. **Accommodation Recommendations** for different budgets
3. **Transportation Options** between locations
4. **Restaurant Suggestions** for local cuisine
5. **Activity Bookings** and reservation tips
6. **Budget Breakdown** with cost estimates
7. **Packing List** recommendations
8. **Emergency Contacts** and important information
9. **Backup Plans** for weather or other contingencies

Make the itinerary practical, realistic, and tailored to the user's interests and budget.""",
            variables=['context', 'query', 'trip_details', 'duration', 'budget', 'interests', 'group_composition'],
            description="Template for creating travel itineraries",
            prompt_type=PromptType.PLANNING
        )
        
        # Chain of Thought Template
        templates['chain_of_thought'] = PromptTemplate(
            name="chain_of_thought",
            template="""You are an expert travel advisor. Think through this travel query step by step.

Context: {context}
Query: {query}

Let's think through this step by step:

1. First, let me understand what the user is asking for...
2. Based on the context, I can see that...
3. Considering the user's needs, I should recommend...
4. The best approach would be...
5. Here's my detailed recommendation:

[Provide your response here]""",
            variables=['context', 'query'],
            description="Template for chain-of-thought reasoning",
            prompt_type=PromptType.GENERAL
        )
        
        return templates
    
    def _initialize_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Initialize few-shot learning examples."""
        return {
            'recommendation': [
                {
                    'query': 'I want to visit a city with great food and culture',
                    'response': 'Based on your interest in food and culture, I recommend Tokyo. Tokyo offers incredible cuisine from street food to Michelin-starred restaurants, plus rich cultural experiences like traditional temples and modern art galleries.'
                },
                {
                    'query': 'Looking for a romantic destination in Europe',
                    'response': 'For a romantic European getaway, I recommend Paris. Known as the City of Light, Paris offers charming cafes, beautiful architecture, romantic Seine River cruises, and world-class dining perfect for couples.'
                }
            ],
            'comparison': [
                {
                    'query': 'Compare Paris vs London for a family trip',
                    'response': 'Paris vs London for families: Paris offers iconic landmarks like the Eiffel Tower and Disneyland Paris, while London has the London Eye and Harry Potter attractions. London has better public transport, Paris has more cultural sites. Both are family-friendly with excellent museums.'
                }
            ],
            'planning': [
                {
                    'query': 'Plan a 5-day trip to Japan',
                    'response': '5-day Japan itinerary: Day 1-2: Tokyo (temples, food, shopping), Day 3: Day trip to Nikko or Kamakura, Day 4: Travel to Kyoto (temples, gardens), Day 5: Return to Tokyo for departure. Include JR Pass for transportation.'
                }
            ]
        }
    
    def _initialize_chain_of_thought_examples(self) -> List[Dict]:
        """Initialize chain-of-thought examples."""
        return [
            {
                'query': 'What\'s the best time to visit Southeast Asia?',
                'reasoning': 'Let me think about this step by step. Southeast Asia has different climates across countries. Thailand, Vietnam, and Cambodia have wet and dry seasons. The dry season (November to April) is generally best for travel. However, I should consider specific countries and activities the user is interested in.',
                'response': 'The best time to visit Southeast Asia is during the dry season from November to April. This period offers pleasant weather, less rainfall, and ideal conditions for beach activities and temple visits.'
            }
        ]
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with provided variables."""
        template = self.get_template(template_name)
        
        # Check if all required variables are provided
        missing_vars = [var for var in template.variables if var not in kwargs]
        if missing_vars:
            # Fill missing variables with defaults
            for var in missing_vars:
                kwargs[var] = self._get_default_value(var)
        
        return template.template.format(**kwargs)
    
    def _get_default_value(self, variable: str) -> str:
        """Get default value for missing variables."""
        defaults = {
            'context': 'No specific context provided',
            'preferences': 'No specific preferences mentioned',
            'destination': 'Not specified',
            'budget': 'Not specified',
            'travel_style': 'Not specified',
            'group_size': 'Not specified',
            'info_type': 'General information',
            'user_level': 'Intermediate',
            'criteria': 'General comparison',
            'trip_details': 'Not specified',
            'duration': 'Not specified',
            'interests': 'Not specified',
            'group_composition': 'Not specified'
        }
        return defaults.get(variable, 'Not specified')
    
    def create_few_shot_prompt(self, query: str, context: str, prompt_type: str = 'recommendation') -> str:
        """Create a few-shot learning prompt."""
        if prompt_type not in self.few_shot_examples:
            prompt_type = 'recommendation'
        
        examples = self.few_shot_examples[prompt_type]
        
        prompt = f"""You are an expert travel advisor. Here are some examples of how to respond to travel queries:

"""
        
        for i, example in enumerate(examples, 1):
            prompt += f"""Example {i}:
Query: {example['query']}
Response: {example['response']}

"""
        
        prompt += f"""Now, based on the following context and examples, respond to this query:

Context: {context}
Query: {query}

Response:"""
        
        return prompt
    
    def create_chain_of_thought_prompt(self, query: str, context: str) -> str:
        """Create a chain-of-thought prompt."""
        example = random.choice(self.chain_of_thought_examples)
        
        prompt = f"""You are an expert travel advisor. Here's an example of how to think through travel queries step by step:

Example:
Query: {example['query']}
Reasoning: {example['reasoning']}
Response: {example['response']}

Now, think through this query step by step:

Context: {context}
Query: {query}

Let's think through this step by step:

1. First, let me understand what the user is asking for...
2. Based on the context, I can see that...
3. Considering the user's needs, I should recommend...
4. The best approach would be...
5. Here's my detailed recommendation:

[Provide your response here]"""
        
        return prompt
    
    def create_dynamic_prompt(self, user_id: str, query: str, context: str) -> str:
        """Create a dynamic prompt based on user profile and history."""
        user_profile = self.user_profiles.get(user_id, {})
        
        # Build dynamic prompt based on user profile
        prompt_parts = [
            "You are an expert travel advisor with deep knowledge of destinations worldwide.",
            f"User Query: {query}",
            f"Context: {context}"
        ]
        
        # Add user-specific information
        if user_profile:
            prompt_parts.append("User Profile Information:")
            
            if 'travel_style' in user_profile:
                prompt_parts.append(f"- Travel Style: {user_profile['travel_style']}")
            
            if 'budget_range' in user_profile:
                prompt_parts.append(f"- Budget Range: {user_profile['budget_range']}")
            
            if 'interests' in user_profile:
                prompt_parts.append(f"- Interests: {', '.join(user_profile['interests'])}")
            
            if 'previous_destinations' in user_profile:
                prompt_parts.append(f"- Previously Visited: {', '.join(user_profile['previous_destinations'])}")
            
            if 'preferences' in user_profile:
                prompt_parts.append(f"- Preferences: {user_profile['preferences']}")
        
        # Add instruction
        prompt_parts.append("""
Please provide a personalized response that takes into account the user's profile and preferences. 
Be specific, helpful, and engaging in your recommendations.""")
        
        return "\n".join(prompt_parts)
    
    def optimize_prompt(self, prompt: str, optimization_type: str = 'clarity') -> str:
        """Optimize a prompt for better performance."""
        if optimization_type == 'clarity':
            return self._optimize_for_clarity(prompt)
        elif optimization_type == 'specificity':
            return self._optimize_for_specificity(prompt)
        elif optimization_type == 'engagement':
            return self._optimize_for_engagement(prompt)
        elif optimization_type == 'accuracy':
            return self._optimize_for_accuracy(prompt)
        else:
            return prompt
    
    def _optimize_for_clarity(self, prompt: str) -> str:
        """Optimize prompt for clarity."""
        optimized = f"""Please provide a clear, well-structured response to the following query.

{prompt}

Structure your response with:
- Clear headings and sections
- Bullet points for lists
- Specific details and examples
- Actionable advice and recommendations
- Easy-to-follow format"""
        
        return optimized
    
    def _optimize_for_specificity(self, prompt: str) -> str:
        """Optimize prompt for specificity."""
        optimized = f"""Provide specific, detailed information in your response.

{prompt}

Include:
- Specific names of places, restaurants, attractions
- Exact prices and costs when possible
- Precise dates and times
- Detailed descriptions and explanations
- Concrete examples and recommendations"""
        
        return optimized
    
    def _optimize_for_engagement(self, prompt: str) -> str:
        """Optimize prompt for user engagement."""
        optimized = f"""Provide an engaging, conversational response that makes the user excited about travel.

{prompt}

Make your response:
- Enthusiastic and inspiring
- Personal and relatable
- Include interesting facts and stories
- Use engaging language and tone
- End with encouraging next steps"""
        
        return optimized
    
    def _optimize_for_accuracy(self, prompt: str) -> str:
        """Optimize prompt for accuracy."""
        optimized = f"""Provide accurate, fact-checked information in your response.

{prompt}

Ensure your response:
- Contains verified information
- Includes current data and facts
- Acknowledges any limitations or uncertainties
- Provides reliable sources when possible
- Is up-to-date and relevant"""
        
        return optimized
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile for dynamic prompt generation."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        self.user_profiles[user_id].update(profile_data)
    
    def get_prompt_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """Generate variations of a prompt for A/B testing."""
        variations = [base_prompt]
        
        # Variation 1: More formal tone
        formal_prompt = base_prompt.replace("You are an expert travel advisor", 
                                          "You are a professional travel consultant")
        variations.append(formal_prompt)
        
        # Variation 2: More casual tone
        casual_prompt = base_prompt.replace("You are an expert travel advisor", 
                                          "You're a travel expert who loves helping people discover amazing places")
        variations.append(casual_prompt)
        
        # Variation 3: More detailed instructions
        detailed_prompt = base_prompt + "\n\nAdditional Instructions:\n- Be thorough in your response\n- Include practical tips\n- Consider different travel styles"
        variations.append(detailed_prompt)
        
        return variations[:num_variations]
    
    def analyze_prompt_performance(self, prompt: str, response: str, user_feedback: str = None) -> Dict[str, Any]:
        """Analyze the performance of a prompt."""
        analysis = {
            'prompt_length': len(prompt),
            'response_length': len(response),
            'prompt_complexity': self._calculate_complexity(prompt),
            'response_quality': self._assess_response_quality(response),
            'user_satisfaction': None
        }
        
        if user_feedback:
            analysis['user_satisfaction'] = self._parse_user_feedback(user_feedback)
        
        return analysis
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        sentences = text.split('.')
        
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words) if words else 0
        
        complexity = (avg_words_per_sentence * 0.3) + (vocabulary_richness * 0.7)
        return min(complexity, 1.0)
    
    def _assess_response_quality(self, response: str) -> Dict[str, float]:
        """Assess the quality of a response."""
        quality_metrics = {
            'completeness': 0.0,
            'specificity': 0.0,
            'helpfulness': 0.0,
            'engagement': 0.0
        }
        
        # Completeness: Check for key elements
        key_elements = ['recommendation', 'information', 'tip', 'suggestion']
        completeness = sum(1 for element in key_elements if element in response.lower()) / len(key_elements)
        quality_metrics['completeness'] = completeness
        
        # Specificity: Check for specific details
        specific_indicators = ['specific', 'exact', 'precise', 'detailed']
        specificity = sum(1 for indicator in specific_indicators if indicator in response.lower()) / len(specific_indicators)
        quality_metrics['specificity'] = specificity
        
        # Helpfulness: Check for actionable advice
        helpful_indicators = ['should', 'recommend', 'suggest', 'consider', 'try']
        helpfulness = sum(1 for indicator in helpful_indicators if indicator in response.lower()) / len(helpful_indicators)
        quality_metrics['helpfulness'] = helpfulness
        
        # Engagement: Check for engaging language
        engaging_indicators = ['amazing', 'incredible', 'wonderful', 'fantastic', 'exciting']
        engagement = sum(1 for indicator in engaging_indicators if indicator in response.lower()) / len(engaging_indicators)
        quality_metrics['engagement'] = engagement
        
        return quality_metrics
    
    def _parse_user_feedback(self, feedback: str) -> float:
        """Parse user feedback into a satisfaction score."""
        positive_words = ['good', 'great', 'excellent', 'helpful', 'useful', 'perfect']
        negative_words = ['bad', 'terrible', 'useless', 'unhelpful', 'wrong']
        
        feedback_lower = feedback.lower()
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        
        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.2
        else:
            return 0.5

def main():
    """Demonstrate advanced prompt engineering techniques."""
    print("🎯 Advanced Prompt Engineering Demo")
    print("=" * 50)
    
    # Initialize prompt engineer
    prompt_engineer = TravelPromptEngineer()
    
    # Sample query and context
    query = "I want to visit a city with great food and culture for a romantic getaway"
    context = "User is planning a 5-day trip in spring with a mid-range budget"
    
    # 1. Basic template-based prompt
    print("1. Template-based Prompt:")
    basic_prompt = prompt_engineer.format_prompt(
        'recommendation',
        query=query,
        context=context,
        preferences="Romantic, food-focused, cultural experiences",
        destination="Not specified",
        budget="Mid-range",
        travel_style="Romantic",
        group_size="2 people"
    )
    print(f"Prompt: {basic_prompt[:200]}...")
    print()
    
    # 2. Few-shot learning prompt
    print("2. Few-shot Learning Prompt:")
    few_shot_prompt = prompt_engineer.create_few_shot_prompt(query, context, 'recommendation')
    print(f"Prompt: {few_shot_prompt[:200]}...")
    print()
    
    # 3. Chain-of-thought prompt
    print("3. Chain-of-thought Prompt:")
    cot_prompt = prompt_engineer.create_chain_of_thought_prompt(query, context)
    print(f"Prompt: {cot_prompt[:200]}...")
    print()
    
    # 4. Dynamic prompt with user profile
    print("4. Dynamic Prompt with User Profile:")
    prompt_engineer.update_user_profile('user123', {
        'travel_style': 'Romantic',
        'budget_range': '$150-300 per day',
        'interests': ['Food', 'Culture', 'History'],
        'previous_destinations': ['Paris', 'Rome'],
        'preferences': 'Prefers boutique hotels and local experiences'
    })
    
    dynamic_prompt = prompt_engineer.create_dynamic_prompt('user123', query, context)
    print(f"Prompt: {dynamic_prompt[:200]}...")
    print()
    
    # 5. Optimized prompts
    print("5. Optimized Prompts:")
    optimized_prompts = {
        'clarity': prompt_engineer.optimize_prompt(basic_prompt, 'clarity'),
        'specificity': prompt_engineer.optimize_prompt(basic_prompt, 'specificity'),
        'engagement': prompt_engineer.optimize_prompt(basic_prompt, 'engagement')
    }
    
    for opt_type, opt_prompt in optimized_prompts.items():
        print(f"{opt_type.title()}: {opt_prompt[:150]}...")
        print()
    
    # 6. Prompt variations for A/B testing
    print("6. Prompt Variations for A/B Testing:")
    variations = prompt_engineer.get_prompt_variations(basic_prompt, 3)
    for i, variation in enumerate(variations, 1):
        print(f"Variation {i}: {variation[:150]}...")
        print()
    
    # 7. Prompt performance analysis
    print("7. Prompt Performance Analysis:")
    sample_response = "Based on your interest in food and culture for a romantic getaway, I recommend Paris. Paris offers incredible cuisine from street food to Michelin-starred restaurants, plus romantic experiences like Seine River cruises and charming cafes."
    
    analysis = prompt_engineer.analyze_prompt_performance(basic_prompt, sample_response)
    print(f"Analysis Results:")
    for metric, value in analysis.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                print(f"    {sub_metric}: {sub_value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n✅ Advanced Prompt Engineering Demo Complete!")
    print("🎯 Features Demonstrated:")
    print("  - Template-based prompt generation")
    print("  - Few-shot learning prompts")
    print("  - Chain-of-thought reasoning")
    print("  - Dynamic prompt generation with user profiles")
    print("  - Prompt optimization techniques")
    print("  - A/B testing variations")
    print("  - Prompt performance analysis")

if __name__ == "__main__":
    main()
