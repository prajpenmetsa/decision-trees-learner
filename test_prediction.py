import sys
sys.path.append('/home/kushal/Documents/Decision_trees/decision-trees-learner')

try:
    from use_learner_category_model import predict_learner_category
    
    # Test data with lists
    test_data = {
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
    
    print("Testing predict_learner_category function...")
    result = predict_learner_category(test_data, "test topic")
    print("Success! Result:", result)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
