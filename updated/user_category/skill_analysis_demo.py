#!/usr/bin/env python3
"""
Demonstration script for enhanced user categorization with skill analysis
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from use_user_categorization_model import predict_user_category_with_skills, analyze_user_skills

def demo_skill_enhanced_prediction():
    """
    Demonstrate the enhanced prediction with skill analysis
    """
    print("Enhanced User Categorization with Skill Analysis")
    print("=" * 50)
    
    # Example user data with skill information
    sample_user = {
        'obj_score': 0.75,
        'conf_score': 0.4,
        'redo_count': 2,
        'flagged_count': 1,
        'conf_trend': -0.3,
        'purpose': 'scratch',
        'prof_goal': 'intermediate',
        'session': 'long',
        'time_s': 720,
        'skill_level': {'logic': 0.6, 'coding': 0.6, 'memory': 0.5}
    }
    
    print("Sample User Input:")
    print("-" * 20)
    for key, value in sample_user.items():
        print(f"{key}: {value}")
    
    # Get enhanced prediction
    result = predict_user_category_with_skills(**sample_user)
    
    print("\nAnalysis Results:")
    print("-" * 20)
    print(f"User Category: {result['mismatch_pattern']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Base Strategy: {result['personalization_strategy']}")
    
    print(f"\nSkill Analysis:")
    print(f"{result['skill_analysis']['skill_statement']}")
    
    print(f"\nFull Recommendation:")
    print(f"Category: {result['mismatch_pattern']}")
    print(f"Strategy: {result['personalization_strategy']}")
    print(f"Skill Info: {result['skill_analysis']['skill_statement']}")

def demo_skill_analysis_only():
    """
    Demonstrate skill analysis functionality separately
    """
    print("\n" + "=" * 50)
    print("Skill Analysis Examples")
    print("=" * 50)
    
    skill_examples = [
        {'logic': 0.85, 'coding': 0.6, 'memory': 0.7},   # Logic master
        {'logic': 0.5, 'coding': 0.75, 'memory': 0.4},   # Coding master
        {'logic': 0.4, 'coding': 0.5, 'memory': 0.8},    # Memory master
        {'logic': 0.6, 'coding': 0.65, 'memory': 0.6},   # No mastery, coding best
        {'logic': 0.3, 'coding': 0.3, 'memory': 0.4},    # All low, memory best
    ]
    
    for i, skills in enumerate(skill_examples, 1):
        print(f"\nExample {i}: {skills}")
        strongest_skill, skill_score, skill_statement = analyze_user_skills(skills)
        print(f"  {skill_statement}")

if __name__ == "__main__":
    demo_skill_enhanced_prediction()
    demo_skill_analysis_only()
    
    print(f"\n{'=' * 50}")
    print("Key Features Added:")
    print("- Skill analysis with 0.7 mastery threshold")
    print("- Specialized content recommendations per skill")
    print("- Combined strategy integrating both category and skills")
    print("- Backward compatibility with original functions")
