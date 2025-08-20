import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html
import pickle
import numpy as np

# === 1. Define Learning Intervention label to description mapping ===
intervention_descriptions = {
    "Condensed Challenge": "Condensed, challenge-focused modules for fast learners preferring short sessions",
    "Extended Examples": "Extended examples, case studies for fast learners preferring long sessions", 
    "Core Practical": "Core concepts, practical exercises for average learners preferring short sessions",
    "Detailed Varied": "Detailed explanations, varied practice for average learners preferring long sessions",
    "Simple Step-by-Step": "Simple language, step-by-step guides for slow learners preferring short sessions",
    "Multimedia Repetitive": "Multimedia content, repetitive practice for slow learners preferring long sessions",
    "Deeper Conceptual": "Deeper conceptual focus, connection-building for surface learners (overridden to long)",
    "Advanced Depth": "Advanced depth, theoretical connections, application mastery for fast revisiting long sessions",
    "Alternative Explanations": "Alternative explanations, step-by-step breakdown for quick struggling (overridden to long)",
    "Comprehensive Review": "Comprehensive review, multiple learning pathways for average revisiting long sessions",
    "Fundamental Remediation": "Fundamental remediation, micro-concepts for deep struggling (overridden to long)",
    "Multi-modal Support": "Multi-modal extensive support, confidence rebuilding for slow revisiting long sessions",
    "Micro-learning": "Micro-learning, essential concepts only for critical struggling short sessions",
    "Foundation Rebuild": "Comprehensive foundation rebuild, personalized learning path for critical struggling long sessions"
}

# === 2. Create training dataset based on the intervention mapping table ===
training_data = []

# Average module time baseline: 720 seconds (12 minutes)
avg_time = 720

# Helper function to generate time values within ranges
def generate_times_in_range(min_pct, max_pct, count=3):
    min_time = avg_time * min_pct / 100
    max_time = avg_time * max_pct / 100
    return np.linspace(min_time, max_time, count)

# Fast Learners (<60% avg time, <432s)
fast_times = generate_times_in_range(30, 59, 5)  # 216s to 424s

for time_taken in fast_times:
    # Fast Learner - Short Preference
    training_data.append({
        'redo_flag': False,
        'session_preference': 'short',
        'time_taken': time_taken,
        'intervention': 'Condensed Challenge'
    })
    
    # Fast Learner - Long Preference  
    training_data.append({
        'redo_flag': False,
        'session_preference': 'long', 
        'time_taken': time_taken,
        'intervention': 'Extended Examples'
    })

# Average Learners (60-140% avg time, 432-1008s)
avg_times = generate_times_in_range(60, 140, 6)  # 432s to 1008s

for time_taken in avg_times:
    # Average Learner - Short Preference
    training_data.append({
        'redo_flag': False,
        'session_preference': 'short',
        'time_taken': time_taken,
        'intervention': 'Core Practical'
    })
    
    # Average Learner - Long Preference
    training_data.append({
        'redo_flag': False,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Detailed Varied'
    })

# Slow Learners (>140% avg time, >1008s)
slow_times = generate_times_in_range(141, 200, 5)  # 1015s to 1440s

for time_taken in slow_times:
    # Slow Learner - Short Preference
    training_data.append({
        'redo_flag': False,
        'session_preference': 'short',
        'time_taken': time_taken,
        'intervention': 'Simple Step-by-Step'
    })
    
    # Slow Learner - Long Preference
    training_data.append({
        'redo_flag': False,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Multimedia Repetitive'
    })

# === REDO FLAG = TRUE Cases ===

# Surface Learners (Fast Revisitors, <60% avg, <432s)
for time_taken in fast_times:
    # Surface Learner (originally short, overridden to long)
    training_data.append({
        'redo_flag': True,
        'session_preference': 'short',  # Original preference
        'time_taken': time_taken,
        'intervention': 'Deeper Conceptual'  # But gets deeper focus treatment
    })
    
    # Surface Learner - Long Preference
    training_data.append({
        'redo_flag': True,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Advanced Depth'
    })

# Quick Struggling (Average revisitors, 60-100% avg, 432-720s)
quick_struggle_times = generate_times_in_range(60, 100, 4)  # 432s to 720s

for time_taken in quick_struggle_times:
    # Quick Struggling (originally short, overridden to long)
    training_data.append({
        'redo_flag': True,
        'session_preference': 'short',  # Original preference
        'time_taken': time_taken,
        'intervention': 'Alternative Explanations'
    })
    
    # Quick Struggling - Long Preference
    training_data.append({
        'redo_flag': True,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Comprehensive Review'
    })

# Deep Struggling (Slow revisitors, 100-140% avg, 720-1008s)
deep_struggle_times = generate_times_in_range(100, 140, 4)  # 720s to 1008s

for time_taken in deep_struggle_times:
    # Deep Struggling (originally short, overridden to long)
    training_data.append({
        'redo_flag': True,
        'session_preference': 'short',  # Original preference
        'time_taken': time_taken,
        'intervention': 'Fundamental Remediation'
    })
    
    # Deep Struggling - Long Preference
    training_data.append({
        'redo_flag': True,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Multi-modal Support'
    })

# Critical Struggling (Very slow, >140% avg, >1008s)
critical_times = generate_times_in_range(141, 250, 5)  # 1015s to 1800s

for time_taken in critical_times:
    # Critical Struggling - Short Preference (stays short)
    training_data.append({
        'redo_flag': True,
        'session_preference': 'short',
        'time_taken': time_taken,
        'intervention': 'Micro-learning'
    })
    
    # Critical Struggling - Long Preference
    training_data.append({
        'redo_flag': True,
        'session_preference': 'long',
        'time_taken': time_taken,
        'intervention': 'Foundation Rebuild'
    })

# === 3. Add some edge cases and specific examples ===
edge_cases = [
    # Boundary cases for time ranges
    {'redo_flag': False, 'session_preference': 'short', 'time_taken': 431, 'intervention': 'Condensed Challenge'},  # Just under fast
    {'redo_flag': False, 'session_preference': 'long', 'time_taken': 433, 'intervention': 'Detailed Varied'},   # Just over fast
    {'redo_flag': False, 'session_preference': 'short', 'time_taken': 1007, 'intervention': 'Core Practical'},  # Just under slow
    {'redo_flag': False, 'session_preference': 'long', 'time_taken': 1009, 'intervention': 'Multimedia Repetitive'}, # Just over slow
    
    # Edge cases for redo scenarios
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 300, 'intervention': 'Deeper Conceptual'},   # Very fast surface learner
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 400, 'intervention': 'Advanced Depth'},      # Fast long revisiter
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 720, 'intervention': 'Alternative Explanations'}, # Boundary quick struggle
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 1008, 'intervention': 'Comprehensive Review'},     # Boundary comprehensive
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 1500, 'intervention': 'Micro-learning'},    # Critical short
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 2000, 'intervention': 'Foundation Rebuild'},  # Very critical long
]

training_data.extend(edge_cases)

# === 4. Create DataFrame ===
df = pd.DataFrame(training_data)

print(f"Total training examples: {len(df)}")
print(f"Intervention distribution:")
print(df['intervention'].value_counts())

# === 5. Prepare features for training ===
# Encode categorical variables
le_session = LabelEncoder()
le_intervention = LabelEncoder()

df['redo_flag_encoded'] = df['redo_flag'].astype(int)  # Boolean to int
df['session_preference_encoded'] = le_session.fit_transform(df['session_preference'])
df['intervention_encoded'] = le_intervention.fit_transform(df['intervention'])

# === 6. Define features and target ===
features = ['redo_flag_encoded', 'session_preference_encoded', 'time_taken']
X = df[features]
y = df['intervention_encoded']

# === 7. Train the decision tree model ===
intervention_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
intervention_model.fit(X, y)

# === 8. Visualize using Graphviz ===
escaped_intervention_labels = [html.escape(label) for label in le_intervention.classes_]
dot_data = export_graphviz(
    intervention_model,
    out_file=None,
    feature_names=features,
    class_names=escaped_intervention_labels,
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=6  # Limit depth for readability
)

graph = graphviz.Source(dot_data)
graph.render("learning_intervention_decision_tree", format="png", view=True)

# === 9. Save model and encoders ===
with open("learning_intervention_model.pkl", "wb") as f:
    pickle.dump(intervention_model, f)

with open("learning_intervention_encoders.pkl", "wb") as f:
    pickle.dump({
        'session_encoder': le_session,
        'intervention_encoder': le_intervention,
        'intervention_descriptions': intervention_descriptions,
        'features': features,
        'avg_module_time': avg_time
    }, f)

print("Learning Intervention Model and encoders saved successfully!")
print("Decision tree visualization saved as 'learning_intervention_decision_tree.png'")

# === 10. Display model performance ===
from sklearn.metrics import accuracy_score, classification_report

y_pred = intervention_model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Number of training samples: {len(df)}")

# Display feature importance
feature_importance = intervention_model.feature_importances_
print("\nFeature Importance:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

# === 11. Test prediction function ===
def predict_learning_intervention(redo_flag, session_preference, time_taken):
    """
    Predict learning intervention based on input features
    """
    # Encode inputs
    redo_encoded = int(redo_flag)
    session_encoded = le_session.transform([session_preference])[0]
    
    feature_vector = [[redo_encoded, session_encoded, time_taken]]
    prediction_encoded = intervention_model.predict(feature_vector)[0]
    probabilities = intervention_model.predict_proba(feature_vector)[0]
    
    # Decode prediction
    intervention = le_intervention.inverse_transform([prediction_encoded])[0]
    confidence = probabilities[prediction_encoded]
    
    return intervention, confidence

# === 12. Test with representative examples ===
test_cases = [
    # Fast learners (no redo)
    {'redo_flag': False, 'session_preference': 'short', 'time_taken': 350, 'expected': 'Condensed Challenge'},
    {'redo_flag': False, 'session_preference': 'long', 'time_taken': 400, 'expected': 'Extended Examples'},
    
    # Average learners (no redo)
    {'redo_flag': False, 'session_preference': 'short', 'time_taken': 600, 'expected': 'Core Practical'},
    {'redo_flag': False, 'session_preference': 'long', 'time_taken': 800, 'expected': 'Detailed Varied'},
    
    # Slow learners (no redo)
    {'redo_flag': False, 'session_preference': 'short', 'time_taken': 1200, 'expected': 'Simple Step-by-Step'},
    {'redo_flag': False, 'session_preference': 'long', 'time_taken': 1400, 'expected': 'Multimedia Repetitive'},
    
    # Surface learners (redo, fast)
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 350, 'expected': 'Deeper Conceptual'},
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 400, 'expected': 'Advanced Depth'},
    
    # Quick struggling (redo, average)
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 650, 'expected': 'Alternative Explanations'},
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 700, 'expected': 'Comprehensive Review'},
    
    # Deep struggling (redo, slow-ish)
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 900, 'expected': 'Fundamental Remediation'},
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 950, 'expected': 'Multi-modal Support'},
    
    # Critical struggling (redo, very slow)
    {'redo_flag': True, 'session_preference': 'short', 'time_taken': 1500, 'expected': 'Micro-learning'},
    {'redo_flag': True, 'session_preference': 'long', 'time_taken': 1600, 'expected': 'Foundation Rebuild'},
]

print(f"\n{'='*80}")
print("Testing Learning Intervention Model")
print("=" * 80)

correct_predictions = 0
for i, test_case in enumerate(test_cases, 1):
    redo_flag = test_case['redo_flag']
    session_pref = test_case['session_preference']
    time_taken = test_case['time_taken']
    expected = test_case.get('expected', 'Unknown')
    
    prediction, confidence = predict_learning_intervention(redo_flag, session_pref, time_taken)
    
    is_correct = "✓" if prediction == expected else "✗"
    if prediction == expected:
        correct_predictions += 1
    
    print(f"Test {i:2d}: redo={redo_flag}, session={session_pref}, time={time_taken}s")
    print(f"         Expected: {expected}")
    print(f"         Predicted: {prediction} (confidence: {confidence:.3f}) {is_correct}")
    if prediction in intervention_descriptions:
        print(f"         Description: {intervention_descriptions[prediction]}")
    print()

print(f"Test Accuracy: {correct_predictions}/{len(test_cases)} = {correct_predictions/len(test_cases):.3f}")

# === 13. Display time range analysis ===
print(f"\n{'='*80}")
print("Time Range Analysis (based on 720s average)")
print("=" * 80)

time_ranges = [
    ("Fast (<60%)", 0, 432),
    ("Average (60-140%)", 432, 1008), 
    ("Slow (>140%)", 1008, float('inf'))
]

for range_name, min_time, max_time in time_ranges:
    if max_time == float('inf'):
        range_data = df[df['time_taken'] > min_time]
    else:
        range_data = df[(df['time_taken'] >= min_time) & (df['time_taken'] <= max_time)]
    
    total = len(range_data)
    redo_count = len(range_data[range_data['redo_flag'] == True])
    no_redo_count = len(range_data[range_data['redo_flag'] == False])
    
    if total > 0:
        print(f"{range_name}: {total} samples")
        print(f"  - No Redo: {no_redo_count} ({no_redo_count/total*100:.1f}%)")
        print(f"  - With Redo: {redo_count} ({redo_count/total*100:.1f}%)")
        
        # Show intervention distribution for this range
        interventions = range_data['intervention'].value_counts()
        print("  - Top Interventions:")
        for intervention, count in interventions.head(3).items():
            print(f"    * {intervention}: {count}")
        print()
    else:
        print(f"{range_name}: No samples")
