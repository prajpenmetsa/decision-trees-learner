#!/usr/bin/env python3

# Test script for the updated combined learner system
import sys
import traceback

try:
    print("Testing the updated combined learner system...")
    
    # Import the CombinedLearnerSystem
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
    
    # Print results
    system.print_formatted_result(result)
    
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
