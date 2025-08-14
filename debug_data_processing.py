#!/usr/bin/env python3

import pickle
import pandas as pd

# Load encoders to check features
try:
    with open("learner_category_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        print("Features expected by model:", encoders['features'])
        print("Encoder keys:", list(encoders.keys()))
except Exception as e:
    print(f"Error loading encoders: {e}")

# Test data processing
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

print("\nOriginal data:")
for key, value in test_data.items():
    print(f"  {key}: {value} (type: {type(value)})")

# Process data like in the function
processed_data = test_data.copy()

# Handle flagged_topics
if isinstance(processed_data.get('flagged_topics'), list):
    processed_data['flagged_topics'] = len(processed_data['flagged_topics'])

# Handle redo_topics
if isinstance(processed_data.get('redo_topics'), list):
    processed_data['redo_topics'] = len(processed_data['redo_topics'])

print("\nProcessed data:")
for key, value in processed_data.items():
    print(f"  {key}: {value} (type: {type(value)})")

# Try creating DataFrame
try:
    input_df = pd.DataFrame([processed_data])
    print("\n✅ DataFrame created successfully")
    print("DataFrame shape:", input_df.shape)
    print("DataFrame columns:", list(input_df.columns))
    print("DataFrame dtypes:")
    print(input_df.dtypes)
except Exception as e:
    print(f"\n❌ Error creating DataFrame: {e}")
