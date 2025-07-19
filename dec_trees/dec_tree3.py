import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html
import pickle
import numpy as np

# === 1. Define comprehensive learner data with mutually exclusive categories ===

learner_data = [
    # Struggling Novice: avg_objective_score < 5.5 AND avg_confidence_score < 0.4 AND learner_level = basic
    {'avg_objective_score': 4.5, 'avg_confidence_score': 0.3, 'confidence_trend': -0.1, 'avg_skill_score': 3.5, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 3, 'redo_topics': 4, 'stddev_objective_score': 0.5, 'category': 'Struggling Novice'},
    {'avg_objective_score': 4.2, 'avg_confidence_score': 0.35, 'confidence_trend': -0.05, 'avg_skill_score': 3.2, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 4, 'redo_topics': 5, 'stddev_objective_score': 0.6, 'category': 'Struggling Novice'},
    
    # Hesitant Learner: avg_objective_score < 5.5 AND avg_confidence_score >= 0.4 AND avg_confidence_score <= 0.6
    {'avg_objective_score': 4.8, 'avg_confidence_score': 0.45, 'confidence_trend': 0.0, 'avg_skill_score': 3.8, 'learner_level': 'intermediate', 'learner_purpose': 'scratch', 'flagged_topics': 2, 'redo_topics': 2, 'stddev_objective_score': 0.4, 'category': 'Hesitant Learner'},
    {'avg_objective_score': 5.2, 'avg_confidence_score': 0.55, 'confidence_trend': -0.02, 'avg_skill_score': 4.0, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 3, 'redo_topics': 3, 'stddev_objective_score': 0.5, 'category': 'Hesitant Learner'},
    
    # Overconfident Novice: avg_objective_score < 5.5 AND avg_confidence_score > 0.6
    {'avg_objective_score': 4.9, 'avg_confidence_score': 0.7, 'confidence_trend': -0.1, 'avg_skill_score': 3.2, 'learner_level': 'basic', 'learner_purpose': 'scratch', 'flagged_topics': 1, 'redo_topics': 1, 'stddev_objective_score': 0.3, 'category': 'Overconfident Novice'},
    {'avg_objective_score': 5.0, 'avg_confidence_score': 0.75, 'confidence_trend': 0.05, 'avg_skill_score': 3.5, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 2, 'redo_topics': 2, 'stddev_objective_score': 0.4, 'category': 'Overconfident Novice'},
    
    # Anxious Improver: avg_objective_score >= 5.5 AND avg_objective_score < 6.5 AND avg_confidence_score < 0.4
    {'avg_objective_score': 5.8, 'avg_confidence_score': 0.35, 'confidence_trend': -0.08, 'avg_skill_score': 4.2, 'learner_level': 'intermediate', 'learner_purpose': 'scratch', 'flagged_topics': 2, 'redo_topics': 1, 'stddev_objective_score': 0.7, 'category': 'Anxious Improver'},
    {'avg_objective_score': 6.2, 'avg_confidence_score': 0.38, 'confidence_trend': -0.05, 'avg_skill_score': 4.5, 'learner_level': 'intermediate', 'learner_purpose': 'revising', 'flagged_topics': 1, 'redo_topics': 2, 'stddev_objective_score': 0.5, 'category': 'Anxious Improver'},
    
    # Rising Improver: avg_objective_score >= 5.5 AND avg_objective_score < 6.5 AND avg_confidence_score >= 0.4 AND avg_confidence_score <= 0.6
    {'avg_objective_score': 6.0, 'avg_confidence_score': 0.55, 'confidence_trend': 0.08, 'avg_skill_score': 4.5, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.6, 'category': 'Rising Improver'},
    {'avg_objective_score': 5.9, 'avg_confidence_score': 0.48, 'confidence_trend': 0.06, 'avg_skill_score': 4.3, 'learner_level': 'intermediate', 'learner_purpose': 'scratch', 'flagged_topics': 1, 'redo_topics': 1, 'stddev_objective_score': 0.4, 'category': 'Rising Improver'},
    
    # Overreacher: avg_objective_score >= 5.5 AND avg_objective_score < 6.5 AND avg_confidence_score > 0.6
    {'avg_objective_score': 6.1, 'avg_confidence_score': 0.72, 'confidence_trend': 0.02, 'avg_skill_score': 4.0, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 2, 'redo_topics': 1, 'stddev_objective_score': 0.8, 'category': 'Overreacher'},
    {'avg_objective_score': 5.7, 'avg_confidence_score': 0.68, 'confidence_trend': 0.04, 'avg_skill_score': 3.8, 'learner_level': 'basic', 'learner_purpose': 'exploratory', 'flagged_topics': 3, 'redo_topics': 2, 'stddev_objective_score': 0.9, 'category': 'Overreacher'},
    
    # Confidence-Delayed: avg_objective_score >= 6.5 AND avg_objective_score < 8.0 AND avg_confidence_score < 0.4
    {'avg_objective_score': 7.0, 'avg_confidence_score': 0.35, 'confidence_trend': -0.12, 'avg_skill_score': 5.6, 'learner_level': 'intermediate', 'learner_purpose': 'revising', 'flagged_topics': 2, 'redo_topics': 3, 'stddev_objective_score': 0.9, 'category': 'Confidence-Delayed'},
    {'avg_objective_score': 6.8, 'avg_confidence_score': 0.38, 'confidence_trend': -0.08, 'avg_skill_score': 5.2, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 1, 'redo_topics': 2, 'stddev_objective_score': 0.6, 'category': 'Confidence-Delayed'},
    
    # Steady Performer: avg_objective_score >= 6.5 AND avg_objective_score < 8.0 AND avg_confidence_score >= 0.4 AND avg_confidence_score <= 0.6
    {'avg_objective_score': 6.5, 'avg_confidence_score': 0.55, 'confidence_trend': 0.01, 'avg_skill_score': 5.0, 'learner_level': 'intermediate', 'learner_purpose': 'revising', 'flagged_topics': 1, 'redo_topics': 0, 'stddev_objective_score': 0.4, 'category': 'Steady Performer'},
    {'avg_objective_score': 7.3, 'avg_confidence_score': 0.52, 'confidence_trend': 0.02, 'avg_skill_score': 5.5, 'learner_level': 'intermediate', 'learner_purpose': 'scratch', 'flagged_topics': 0, 'redo_topics': 1, 'stddev_objective_score': 0.3, 'category': 'Steady Performer'},
    
    # Confident Achiever: avg_objective_score >= 6.5 AND avg_objective_score < 8.0 AND avg_confidence_score > 0.6
    {'avg_objective_score': 6.8, 'avg_confidence_score': 0.7, 'confidence_trend': 0.07, 'avg_skill_score': 4.2, 'learner_level': 'intermediate', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.5, 'category': 'Confident Achiever'},
    {'avg_objective_score': 7.5, 'avg_confidence_score': 0.68, 'confidence_trend': 0.05, 'avg_skill_score': 5.8, 'learner_level': 'advanced', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.4, 'category': 'Confident Achiever'},
    
    # Humble Expert: avg_objective_score >= 8.0 AND avg_confidence_score >= 0.4 AND avg_confidence_score <= 0.6
    {'avg_objective_score': 8.2, 'avg_confidence_score': 0.55, 'confidence_trend': 0.03, 'avg_skill_score': 6.5, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.2, 'category': 'Humble Expert'},
    {'avg_objective_score': 8.0, 'avg_confidence_score': 0.48, 'confidence_trend': 0.01, 'avg_skill_score': 6.2, 'learner_level': 'advanced', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.3, 'category': 'Humble Expert'},
    
    # Stable Expert: avg_objective_score >= 8.0 AND avg_confidence_score > 0.6
    {'avg_objective_score': 8.5, 'avg_confidence_score': 0.75, 'confidence_trend': 0.1, 'avg_skill_score': 6.2, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.2, 'category': 'Stable Expert'},
    {'avg_objective_score': 8.8, 'avg_confidence_score': 0.82, 'confidence_trend': 0.05, 'avg_skill_score': 6.8, 'learner_level': 'advanced', 'learner_purpose': 'exploratory', 'flagged_topics': 0, 'redo_topics': 0, 'stddev_objective_score': 0.1, 'category': 'Stable Expert'},
    
    # Imposter Syndrome: avg_objective_score >= 8.0 AND avg_confidence_score < 0.4
    {'avg_objective_score': 8.3, 'avg_confidence_score': 0.35, 'confidence_trend': -0.06, 'avg_skill_score': 6.5, 'learner_level': 'advanced', 'learner_purpose': 'revising', 'flagged_topics': 1, 'redo_topics': 0, 'stddev_objective_score': 0.4, 'category': 'Imposter Syndrome'},
    {'avg_objective_score': 8.1, 'avg_confidence_score': 0.32, 'confidence_trend': -0.08, 'avg_skill_score': 6.2, 'learner_level': 'advanced', 'learner_purpose': 'scratch', 'flagged_topics': 2, 'redo_topics': 1, 'stddev_objective_score': 0.5, 'category': 'Imposter Syndrome'},
]

# === 2. Create DataFrame and calculate redo_topics_flag ===
df = pd.DataFrame(learner_data)

# Calculate redo_topics_flag based on performance level and existing redo_topics
def calculate_redo_flag(row):
    score = row['avg_objective_score']
    existing_redo = row['redo_topics']
    
    if score < 5.5:  # Struggling
        return True
    elif 5.5 <= score < 6.5:  # Developing
        return existing_redo >= 2
    elif 6.5 <= score < 8.0:  # Competent
        return existing_redo >= 3
    else:  # Expert
        return False

df['redo_topics_flag'] = df.apply(calculate_redo_flag, axis=1)

# === 3. Define prompt mappings ===
category_prompts = {
    'Struggling Novice': "Provide beginner-friendly content for {next_topic} using simple language and clear examples. Break concepts into small steps with frequent encouragement. Build confidence through achievable milestones.",
    'Hesitant Learner': "Teach {next_topic} with gentle scaffolding and reassurance. Provide clear structure and validate every small progress. Address uncertainty with supportive feedback.",
    'Overconfident Novice': "Create corrective content for {next_topic} using misconception targeting. Gently challenge assumptions with evidence-based examples while maintaining motivation.",
    'Anxious Improver': "Build confidence in {next_topic} through success recognition. Emphasize progress made and provide reassuring guidance. Use positive reinforcement to overcome self-doubt.",
    'Rising Improver': "Encourage steady progress in {next_topic} with moderate challenges. Use supportive feedback and clear progress indicators. Celebrate improvements while maintaining momentum.",
    'Overreacher': "Provide appropriately challenging content for {next_topic}. Include reality checks and just-in-time hints to prevent overextension. Guide ambition with structured support.",
    'Confidence-Delayed': "Encourage self-trust in {next_topic} by highlighting existing competence. Reduce hand-holding to build independent confidence. Emphasize capability over support.",
    'Steady Performer': "Provide standard-paced content for {next_topic} with moderate challenges. Include brief concept checks and maintain consistent progression. Focus on steady skill building.",
    'Confident Achiever': "Deliver engaging content for {next_topic} with varied challenges. Encourage exploration and practical application. Provide opportunities for creative problem-solving.",
    'Humble Expert': "Present advanced content for {next_topic} with nuanced discussions. Acknowledge expertise while encouraging continued growth. Focus on mastery refinement.",
    'Stable Expert': "Generate concise, expert-level content for {next_topic}. Focus on advanced applications and cutting-edge concepts. Minimal guidance needed - encourage innovation.",
    'Imposter Syndrome': "Affirm expertise while teaching {next_topic}. Provide evidence of competence and reduce self-doubt. Focus on recognizing mastery already achieved."
}

# === 4. Encode categorical variables ===
le_level = LabelEncoder().fit(df['learner_level'])
le_purpose = LabelEncoder().fit(df['learner_purpose'])
le_category = LabelEncoder().fit(df['category'])

df['learner_level_encoded'] = le_level.transform(df['learner_level'])
df['learner_purpose_encoded'] = le_purpose.transform(df['learner_purpose'])
df['category_encoded'] = le_category.transform(df['category'])

# === 5. Prepare data for training ===
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

# === 6. Train the decision tree model ===
model = DecisionTreeClassifier(
    max_depth=10,  # Prevent overfitting
    min_samples_split=2,  # Minimum samples to split
    min_samples_leaf=1,   # Minimum samples in leaf
    random_state=42
)
model.fit(X, y)

# === 7. Visualize using Graphviz ===
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=features,
    class_names=[html.escape(label) for label in le_category.classes_],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=5  # Limit visualization depth for readability
)

graph = graphviz.Source(dot_data)
graph.render("comprehensive_learner_decision_tree", format="png", view=True)

# === 8. Enhanced prediction function with prompt generation ===
def predict_learner_profile(new_data, next_topic="the current topic"):
    """
    Predict learner category and generate appropriate prompt
    """
    # Create DataFrame with a single row
    input_df = pd.DataFrame([new_data])
    
    # Calculate redo_topics_flag
    input_df['redo_topics_flag'] = input_df.apply(calculate_redo_flag, axis=1)
    
    # Encode categorical variables
    if 'learner_level' in input_df.columns:
        input_df['learner_level_encoded'] = le_level.transform(input_df['learner_level'])
    
    if 'learner_purpose' in input_df.columns:
        input_df['learner_purpose_encoded'] = le_purpose.transform(input_df['learner_purpose'])
    
    # Extract features
    X_input = input_df[features]
    
    # Make prediction
    predicted_category_encoded = model.predict(X_input)[0]
    predicted_category = le_category.inverse_transform([predicted_category_encoded])[0]
    
    # Generate prompt
    base_prompt = category_prompts[predicted_category]
    final_prompt = base_prompt.format(next_topic=next_topic)
    
    # Add redo context if needed
    redo_flag = input_df['redo_topics_flag'].iloc[0]
    if redo_flag and new_data.get('redo_topics', 0) > 0:
        final_prompt += f" Focus on reviewing and reinforcing previous topics that need attention."
    
    return {
        'category': predicted_category,
        'redo_topics_flag': redo_flag,
        'prompt': final_prompt,
        'confidence': max(model.predict_proba(X_input)[0])
    }

# === 9. Example usage ===
new_learner = {
    'avg_objective_score': 10, 
    'avg_confidence_score': 8, 
    'confidence_trend': 0.05, 
    'avg_skill_score': 4.8, 
    'learner_level': 'intermediate', 
    'learner_purpose': 'exploratory', 
    'flagged_topics': 1, 
    'redo_topics': 0, 
    'stddev_objective_score': 0.4
}

# Get comprehensive prediction
result = predict_learner_profile(new_learner, "machine learning fundamentals")
print(f"Predicted Category: {result['category']}")
print(f"Redo Topics Flag: {result['redo_topics_flag']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Generated Prompt: {result['prompt']}")

# === 10. Model evaluation ===
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=le_category.classes_))

print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# === 11. Save all components ===
# Save model and encoders
with open("comprehensive_learner_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("learner_encoders.pkl", "wb") as f:
    pickle.dump({
        'level_encoder': le_level,
        'purpose_encoder': le_purpose,
        'category_encoder': le_category,
        'category_prompts': category_prompts
    }, f)

print("\nModel and encoders saved successfully!")
print("Decision tree visualization saved as 'comprehensive_learner_decision_tree.png'")

# === Separate Decision Tree for Topic Revisit Logic ===
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

# Define data for topic revisit logic
revisit_data = [
    {'avg_objective_score': 5.0, 'redo_topics': 4, 'redo_flag': True},  # Struggling
    {'avg_objective_score': 6.0, 'redo_topics': 2, 'redo_flag': True},  # Developing
    {'avg_objective_score': 7.0, 'redo_topics': 3, 'redo_flag': True},  # Competent
    {'avg_objective_score': 8.5, 'redo_topics': 0, 'redo_flag': False}  # Expert
]

# Create DataFrame
revisit_df = pd.DataFrame(revisit_data)

# Features and target
revisit_features = ['avg_objective_score', 'redo_topics']
revisit_target = 'redo_flag'

# Train decision tree for topic revisit logic
revisit_model = DecisionTreeClassifier(max_depth=3, random_state=42)
revisit_model.fit(revisit_df[revisit_features], revisit_df[revisit_target])

# Save the revisit model
with open("topic_revisit_decision_tree.pkl", "wb") as f:
    pickle.dump(revisit_model, f)

# Function to predict topic revisit
def predict_topic_revisit(avg_objective_score, redo_topics):
    """
    Predict if a learner needs to revisit topics based on avg_objective_score and redo_topics.

    Args:
        avg_objective_score (float): Average objective score.
        redo_topics (int): Number of redo topics.

    Returns:
        bool: True if the learner needs to revisit topics, False otherwise.
    """
    input_data = pd.DataFrame([{"avg_objective_score": avg_objective_score, "redo_topics": redo_topics}])
    return revisit_model.predict(input_data)[0]

# Example usage
if __name__ == "__main__":
    avg_objective_score = 8
    redo_topics = 2
    needs_revisit = predict_topic_revisit(avg_objective_score, redo_topics)
    print(f"Does the learner need to revisit topics? {'Yes' if needs_revisit else 'No'}")