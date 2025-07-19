import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html
import pickle

# === 1. Define your decision logic categories (from the CART rules you finalized) ===

learner_data = [
    {'avg_objective_score': 4.5, 'avg_confidence_score': 0.3, 'confidence_trend': -0.1, 'avg_skill_score': 3.5, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 3, 'redo_topics': 4, 'stddev_objective_score': 0.5, 'category': 'Struggling Novice'},
    {'avg_objective_score': 4.8, 'avg_confidence_score': 0.35, 'confidence_trend': 0.0, 'avg_skill_score': 3.8, 'learner_level': 'intermediate', 'learner_purpose': 'scratch', 'flagged_topics': 2, 'redo_topics': 2, 'stddev_objective_score': 0.4, 'category': 'Lost Climber'},
    {'avg_objective_score': 4.9, 'avg_confidence_score': 0.7, 'confidence_trend': -0.1, 'avg_skill_score': 3.2, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 1, 'redo_topics': 1, 'stddev_objective_score': 0.3, 'category': 'Confused Confident'},
    {'avg_objective_score': 6.0, 'avg_confidence_score': 0.55, 'confidence_trend': 0.08, 'avg_skill_score': 4.5, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.6, 'category': 'Rising Improver'},
    {'avg_objective_score': 6.5, 'avg_confidence_score': 0.55, 'confidence_trend': 0.01, 'avg_skill_score': 5.0, 'learner_level': 'intermediate', 'learner_purpose': 'revising', 'flagged_topics': 1, 'redo_topics': 0, 'stddev_objective_score': 0.4, 'category': 'Stabilized Climber'},
    {'avg_objective_score': 7.2, 'avg_confidence_score': 0.65, 'confidence_trend': 0.0, 'avg_skill_score': 5.8, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.3, 'category': 'Repetition-Focused Pro'},
    {'avg_objective_score': 8.5, 'avg_confidence_score': 0.75, 'confidence_trend': 0.1, 'avg_skill_score': 6.2, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.2, 'category': 'Stable Expert'},
    {'avg_objective_score': 5.9, 'avg_confidence_score': 0.7, 'confidence_trend': 0.02, 'avg_skill_score': 3.5, 'learner_level': 'basic', 'learner_purpose': 'exploratory', 'flagged_topics': 3, 'redo_topics': 1, 'stddev_objective_score': 1.7, 'category': 'Confused Explorer'},
    {'avg_objective_score': 6.8, 'avg_confidence_score': 0.7, 'confidence_trend': 0.07, 'avg_skill_score': 4.2, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.5, 'category': 'Well-Balanced Explorer'},
    {'avg_objective_score': 7.0, 'avg_confidence_score': 0.45, 'confidence_trend': -0.12, 'avg_skill_score': 5.6, 'learner_level': 'intermediate', 'learner_purpose': 'revising', 'flagged_topics': 2, 'redo_topics': 3, 'stddev_objective_score': 0.9, 'category': 'Confidence-Delayed'},
]

df = pd.DataFrame(learner_data)

# === 2. Encode categorical variables ===
le_level = LabelEncoder().fit(df['learner_level'])
le_purpose = LabelEncoder().fit(df['learner_purpose'])
le_category = LabelEncoder().fit(df['category'])

df['learner_level_encoded'] = le_level.transform(df['learner_level'])
df['learner_purpose_encoded'] = le_purpose.transform(df['learner_purpose'])
df['category_encoded'] = le_category.transform(df['category'])

# === 3. Prepare data for training ===
features = [
    'avg_objective_score',
    'avg_confidence_score',
    'confidence_trend',
    'avg_skill_score',
    'learner_level_encoded',
    'learner_purpose_encoded',
    'flagged_topics',
    'redo_topics',
    'stddev_objective_score'
]
X = df[features]
y = df['category_encoded']

# === 4. Train the decision tree model ===
model = DecisionTreeClassifier(max_depth=None, random_state=0)
model.fit(X, y)

# === 5. Visualize using Graphviz ===
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=features,
    class_names=[html.escape(label) for label in le_category.classes_],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("learner_decision_tree", format="png", view=True)  # Saves and opens learner_decision_tree.png

# === 6. Function to predict category for a new learner ===
def predict_learner_category(new_data):
    # Create DataFrame with a single row
    input_df = pd.DataFrame([new_data])
    
    # Encode categorical variables just like we did for training data
    if 'learner_level' in input_df.columns:
        input_df['learner_level_encoded'] = le_level.transform(input_df['learner_level'])
    
    if 'learner_purpose' in input_df.columns:
        input_df['learner_purpose_encoded'] = le_purpose.transform(input_df['learner_purpose'])
    
    # Extract features in the same order used for training
    X_input = input_df[features]
    
    # Make prediction
    predicted_category_encoded = model.predict(X_input)[0]
    
    # Convert encoded prediction back to category name
    predicted_category = le_category.inverse_transform([predicted_category_encoded])[0]
    
    return predicted_category

# Example usage - replace this with your actual input data
new_learner = {
    'avg_objective_score': 6.2, 
    'avg_confidence_score': 0.5, 
    'confidence_trend': 0.05, 
    'avg_skill_score': 4.8, 
    'learner_level': 'intermediate', 
    'learner_purpose': 'exploratory', 
    'flagged_topics': 1, 
    'redo_topics': 0, 
    'stddev_objective_score': 0.4
}

# Get the prediction
predicted_category = predict_learner_category(new_learner)
print(f"Predicted learner category: {predicted_category}")

# Save the trained model to a file
with open("learner_decision_tree_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the encoders to files
with open("learner_level_encoder.pkl", "wb") as level_encoder_file:
    pickle.dump(le_level, level_encoder_file)

with open("learner_purpose_encoder.pkl", "wb") as purpose_encoder_file:
    pickle.dump(le_purpose, purpose_encoder_file)

with open("learner_category_encoder.pkl", "wb") as category_encoder_file:
    pickle.dump(le_category, category_encoder_file)
