#!/usr/bin/env python3

import sys
import traceback
import os

# Change to the correct directory
os.chdir('/home/kushal/Documents/Decision_trees/decision-trees-learner')

try:
    # Import and test the system
    from combined_learner_system import CombinedLearnerSystem
    print("✅ Import successful")
    
    # Initialize the system
    system = CombinedLearnerSystem()
    print("✅ System initialized")
    
    # Test with the new format
    learner_data = {
        'avg_objective_score': 6.0,
        'avg_confidence_score': 0.45,
        'confidence_trend': 0.02,
        'avg_skill_score': 4.5,
        'learner_level': 'intermediate',
        'learner_purpose': 'scratch',
        'flagged_topics': ['variables', 'loops', 'functions'],
        'redo_topics': ['loops', 'conditionals', 'data structures'],
        'stddev_objective_score': 0.6
    }
    
    strategy_data = {
        'session_preference': 'short_chunks',
        'attention_span': 8.5,
        'format_preference': 'text'
    }
    
    print("✅ Test data prepared")
    
    # Make prediction
    result = system.predict_combined(
        learner_data, 
        strategy_data, 
        "Python object-oriented programming"
    )
    
    print("✅ Prediction successful")
    print("Final prompt preview:")
    print(result['combined_output']['final_prompt'][:200] + "...")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
