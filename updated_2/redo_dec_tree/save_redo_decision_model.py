import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import html
import pickle
import numpy as np

# === 1. Define Redo Decision Dataset based on the provided table ===
# Creating comprehensive examples based on the thresholds from the document

redo_data = []

# === 0.0-0.4 Range: Always "YES" (5 examples) ===
for obj_score in [0.1, 0.25, 0.35, 0.4]:
    for redo_count in [0, 2]:
        for flagged_count in [0, 1]:
            if len(redo_data) < 5:  # Limit to 5 examples for this range
                redo_data.append({
                    'obj_score': obj_score,
                    'redo_count': redo_count,
                    'flagged_count': flagged_count,
                    'redo_decision': 'YES'
                })

# === 0.4-0.6 Range: (Redo Count >= 2) OR (Flagged Count >= 2) (12 examples) ===
obj_score = 0.5
for redo_count in range(5):  # 0-4
    for flagged_count in range(4):  # 0-3
        if (redo_count + flagged_count) <= 5:  # Selective sampling
            decision = 'YES' if (redo_count >= 2 or flagged_count >= 2) else 'NO'
            redo_data.append({
                'obj_score': obj_score,
                'redo_count': redo_count,
                'flagged_count': flagged_count,
                'redo_decision': decision
            })

# === 0.6-0.7 Range: (Redo Count >= 3) OR (Flagged Count >= 3) (12 examples) ===
obj_score = 0.65
for redo_count in range(5):  # 0-4
    for flagged_count in range(4):  # 0-3
        if (redo_count + flagged_count) <= 5:  # Selective sampling
            decision = 'YES' if (redo_count >= 3 or flagged_count >= 3) else 'NO'
            redo_data.append({
                'obj_score': obj_score,
                'redo_count': redo_count,
                'flagged_count': flagged_count,
                'redo_decision': decision
            })

# === 0.7-0.8 Range: Complex conditions (12 examples) ===
obj_score = 0.75
for redo_count in range(5):  # 0-4
    for flagged_count in range(4):  # 0-3
        if (redo_count + flagged_count) <= 5:  # Selective sampling
            decision = 'YES' if (redo_count >= 4 or flagged_count >= 4 or 
                               (redo_count >= 3 and flagged_count >= 2)) else 'NO'
            redo_data.append({
                'obj_score': obj_score,
                'redo_count': redo_count,
                'flagged_count': flagged_count,
                'redo_decision': decision
            })

# === 0.8-1.0 Range: Most complex conditions (12 examples) ===
obj_score = 0.9
for redo_count in range(6):  # 0-5
    for flagged_count in range(4):  # 0-3
        if (redo_count + flagged_count) <= 6:  # Selective sampling
            decision = 'YES' if (redo_count >= 5 or flagged_count >= 5 or 
                               (redo_count >= 4 and flagged_count >= 2) or
                               (redo_count >= 3 and flagged_count >= 3)) else 'NO'
            redo_data.append({
                'obj_score': obj_score,
                'redo_count': redo_count,
                'flagged_count': flagged_count,
                'redo_decision': decision
            })

# === Add some specific examples to ensure coverage (8 examples) ===
specific_examples = [
    # Edge cases for 0.4-0.6 range
    {'obj_score': 0.45, 'redo_count': 0, 'flagged_count': 0, 'redo_decision': 'NO'},
    {'obj_score': 0.45, 'redo_count': 2, 'flagged_count': 0, 'redo_decision': 'YES'},
    {'obj_score': 0.45, 'redo_count': 0, 'flagged_count': 2, 'redo_decision': 'YES'},
    
    # Edge cases for 0.6-0.7 range
    {'obj_score': 0.68, 'redo_count': 3, 'flagged_count': 0, 'redo_decision': 'YES'},
    {'obj_score': 0.68, 'redo_count': 0, 'flagged_count': 3, 'redo_decision': 'YES'},
    
    # Edge cases for 0.7-0.8 range
    {'obj_score': 0.78, 'redo_count': 3, 'flagged_count': 2, 'redo_decision': 'YES'},
    {'obj_score': 0.78, 'redo_count': 4, 'flagged_count': 0, 'redo_decision': 'YES'},
    
    # Edge cases for 0.8-1.0 range  
    {'obj_score': 0.95, 'redo_count': 3, 'flagged_count': 3, 'redo_decision': 'YES'},
]

redo_data.extend(specific_examples)

# === 2. Create DataFrame ===
df = pd.DataFrame(redo_data)

# Remove duplicates if any
df = df.drop_duplicates()

print(f"Total training examples: {len(df)}")
print(f"YES decisions: {len(df[df['redo_decision'] == 'YES'])}")
print(f"NO decisions: {len(df[df['redo_decision'] == 'NO'])}")

# === 3. Define features and target ===
features = ['obj_score', 'redo_count', 'flagged_count']
X = df[features]
y = df['redo_decision']

# === 4. Train the decision tree model ===
model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)

# === 5. Visualize using Graphviz ===
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=features,
    class_names=['NO', 'YES'],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=6  # Limit depth for readability
)

graph = graphviz.Source(dot_data)
graph.render("redo_decision_tree", format="png", view=True)

# === 6. Save model and encoders ===
with open("redo_decision_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("redo_decision_encoders.pkl", "wb") as f:
    pickle.dump({
        'features': features,
        'target': 'redo_decision'
    }, f)

print("Redo Decision Model saved successfully!")
print("Decision tree visualization saved as 'redo_decision_tree.png'")

# === 7. Display model performance ===
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on training data
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Number of training samples: {len(df)}")

# Display feature importance
feature_importance = model.feature_importances_
print("\nFeature Importance:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

# === 8. Test prediction function ===
def predict_redo_decision(obj_score, redo_count, flagged_count):
    """
    Predict redo decision based on input features
    """
    feature_vector = [[obj_score, redo_count, flagged_count]]
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0]
    
    # Get probability for the predicted class
    if prediction == 'YES':
        confidence = probability[1]  # Probability of YES
    else:
        confidence = probability[0]  # Probability of NO
    
    return prediction, confidence

# === 9. Test with examples from each range ===
test_cases = [
    # 0.0-0.4 range (should always be YES)
    {'obj_score': 0.3, 'redo_count': 0, 'flagged_count': 0, 'expected': 'YES'},
    {'obj_score': 0.2, 'redo_count': 1, 'flagged_count': 1, 'expected': 'YES'},
    
    # 0.4-0.6 range
    {'obj_score': 0.5, 'redo_count': 0, 'flagged_count': 1, 'expected': 'NO'},
    {'obj_score': 0.5, 'redo_count': 2, 'flagged_count': 0, 'expected': 'YES'},
    {'obj_score': 0.5, 'redo_count': 0, 'flagged_count': 2, 'expected': 'YES'},
    
    # 0.6-0.7 range
    {'obj_score': 0.65, 'redo_count': 2, 'flagged_count': 2, 'expected': 'NO'},
    {'obj_score': 0.65, 'redo_count': 3, 'flagged_count': 0, 'expected': 'YES'},
    {'obj_score': 0.65, 'redo_count': 0, 'flagged_count': 3, 'expected': 'YES'},
    
    # 0.7-0.8 range
    {'obj_score': 0.75, 'redo_count': 3, 'flagged_count': 2, 'expected': 'YES'},
    {'obj_score': 0.75, 'redo_count': 3, 'flagged_count': 1, 'expected': 'NO'},
    {'obj_score': 0.75, 'redo_count': 4, 'flagged_count': 0, 'expected': 'YES'},
    
    # 0.8-1.0 range
    {'obj_score': 0.9, 'redo_count': 3, 'flagged_count': 3, 'expected': 'YES'},
    {'obj_score': 0.9, 'redo_count': 4, 'flagged_count': 2, 'expected': 'YES'},
    {'obj_score': 0.9, 'redo_count': 2, 'flagged_count': 2, 'expected': 'NO'},
]

print(f"\n{'='*60}")
print("Testing Redo Decision Model")
print("=" * 60)

correct_predictions = 0
for i, test_case in enumerate(test_cases, 1):
    obj_score = test_case['obj_score']
    redo_count = test_case['redo_count']
    flagged_count = test_case['flagged_count']
    expected = test_case['expected']
    
    prediction, confidence = predict_redo_decision(obj_score, redo_count, flagged_count)
    
    is_correct = "✓" if prediction == expected else "✗"
    if prediction == expected:
        correct_predictions += 1
    
    print(f"Test {i:2d}: obj={obj_score:.2f}, redo={redo_count}, flagged={flagged_count}")
    print(f"         Expected: {expected}, Predicted: {prediction} ({confidence:.3f}) {is_correct}")

print(f"\nTest Accuracy: {correct_predictions}/{len(test_cases)} = {correct_predictions/len(test_cases):.3f}")

# === 10. Display distribution by score ranges ===
print(f"\n{'='*60}")
print("Distribution by Objective Score Ranges")
print("=" * 60)

ranges = [
    (0.0, 0.4, "0.0-0.4"),
    (0.4, 0.6, "0.4-0.6"), 
    (0.6, 0.7, "0.6-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 1.0, "0.8-1.0")
]

for min_score, max_score, range_name in ranges:
    range_data = df[(df['obj_score'] >= min_score) & (df['obj_score'] <= max_score)]
    yes_count = len(range_data[range_data['redo_decision'] == 'YES'])
    no_count = len(range_data[range_data['redo_decision'] == 'NO'])
    total = len(range_data)
    
    if total > 0:
        yes_pct = yes_count / total * 100
        print(f"{range_name}: {total} samples - YES: {yes_count} ({yes_pct:.1f}%), NO: {no_count} ({100-yes_pct:.1f}%)")
    else:
        print(f"{range_name}: No samples")
