import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html
import pickle
import numpy as np
import random

# === 1. Define User Categorization Dataset based on the provided table ===
user_data = [
    # A1 - Perfect Alignment
    {'obj_score': 0.85, 'conf_score': 0.85, 'redo_count': 0, 'flagged_count': 0, 'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 450, 'mismatch_pattern': 'Perfect Alignment'},
    {'obj_score': 0.9, 'conf_score': 0.9, 'redo_count': 0, 'flagged_count': 0, 'conf_trend': 0.7, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 400, 'mismatch_pattern': 'Perfect Alignment'},
    
    # A2 - Consistent High
    {'obj_score': 0.75, 'conf_score': 0.75, 'redo_count': 1, 'flagged_count': 1, 'conf_trend': -0.1, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 600, 'mismatch_pattern': 'Consistent High'},
    {'obj_score': 0.8, 'conf_score': 0.8, 'redo_count': 2, 'flagged_count': 2, 'conf_trend': 0.1, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 650, 'mismatch_pattern': 'Consistent High'},
    
    # B1 - Impostor Syndrome
    {'obj_score': 0.8, 'conf_score': 0.3, 'redo_count': 1, 'flagged_count': 2, 'conf_trend': -0.5, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 800, 'mismatch_pattern': 'Impostor Syndrome'},
    {'obj_score': 0.85, 'conf_score': 0.25, 'redo_count': 2, 'flagged_count': 1, 'conf_trend': -0.7, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 750, 'mismatch_pattern': 'Impostor Syndrome'},
    
    # B2 - Perfectionist Anxiety
    {'obj_score': 0.75, 'conf_score': 0.2, 'redo_count': 4, 'flagged_count': 3, 'conf_trend': -0.8, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 900, 'mismatch_pattern': 'Perfectionist Anxiety'},
    {'obj_score': 0.8, 'conf_score': 0.15, 'redo_count': 5, 'flagged_count': 4, 'conf_trend': -0.6, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 950, 'mismatch_pattern': 'Perfectionist Anxiety'},
    
    # C1 - Overconfident Underperformer
    {'obj_score': 0.6, 'conf_score': 0.85, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': 0.3, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 400, 'mismatch_pattern': 'Overconfident Underperformer'},
    {'obj_score': 0.55, 'conf_score': 0.9, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': 0.8, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 350, 'mismatch_pattern': 'Overconfident Underperformer'},
    
    # C2 - Confident but Careless
    {'obj_score': 0.6, 'conf_score': 0.8, 'redo_count': 3, 'flagged_count': 1, 'conf_trend': 0.2, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 450, 'mismatch_pattern': 'Confident but Careless'},
    {'obj_score': 0.65, 'conf_score': 0.75, 'redo_count': 2, 'flagged_count': 2, 'conf_trend': -0.3, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 400, 'mismatch_pattern': 'Confident but Careless'},
    
    # D1 - Dunning-Kruger Effect
    {'obj_score': 0.3, 'conf_score': 0.8, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 600, 'mismatch_pattern': 'Dunning-Kruger Effect'},
    {'obj_score': 0.4, 'conf_score': 0.9, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': 0.7, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 650, 'mismatch_pattern': 'Dunning-Kruger Effect'},
    
    # D2 - False Confidence
    {'obj_score': 0.2, 'conf_score': 0.85, 'redo_count': 2, 'flagged_count': 4, 'conf_trend': 0.1, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 400, 'mismatch_pattern': 'False Confidence'},
    {'obj_score': 0.35, 'conf_score': 0.9, 'redo_count': 1, 'flagged_count': 5, 'conf_trend': -0.2, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 450, 'mismatch_pattern': 'False Confidence'},
    
    # E1 - Performance-Practice Disconnect
    {'obj_score': 0.8, 'conf_score': 0.6, 'redo_count': 4, 'flagged_count': 2, 'conf_trend': -0.8, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 850, 'mismatch_pattern': 'Performance-Practice Disconnect'},
    {'obj_score': 0.75, 'conf_score': 0.55, 'redo_count': 5, 'flagged_count': 1, 'conf_trend': -0.6, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 900, 'mismatch_pattern': 'Performance-Practice Disconnect'},
    
    # E2 - Concept Confusion
    {'obj_score': 0.8, 'conf_score': 0.6, 'redo_count': 2, 'flagged_count': 4, 'conf_trend': -0.3, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 550, 'mismatch_pattern': 'Concept Confusion'},
    {'obj_score': 0.85, 'conf_score': 0.65, 'redo_count': 1, 'flagged_count': 5, 'conf_trend': 0.2, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 600, 'mismatch_pattern': 'Concept Confusion'},
    
    # F1 - Analysis Paralysis
    {'obj_score': 0.6, 'conf_score': 0.3, 'redo_count': 1, 'flagged_count': 4, 'conf_trend': -0.7, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 800, 'mismatch_pattern': 'Analysis Paralysis'},
    {'obj_score': 0.65, 'conf_score': 0.4, 'redo_count': 2, 'flagged_count': 5, 'conf_trend': -0.9, 'purpose': 'revision', 'prof_goal': 'basic', 'session': 'short', 'time_s': 850, 'mismatch_pattern': 'Analysis Paralysis'},
    
    # F2 - Practice Avoidance
    {'obj_score': 0.55, 'conf_score': 0.25, 'redo_count': 4, 'flagged_count': 1, 'conf_trend': -0.4, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 650, 'mismatch_pattern': 'Practice Avoidance'},
    {'obj_score': 0.6, 'conf_score': 0.35, 'redo_count': 5, 'flagged_count': 0, 'conf_trend': -0.6, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 700, 'mismatch_pattern': 'Practice Avoidance'},
    
    # G1 - Struggling Accurately
    {'obj_score': 0.3, 'conf_score': 0.25, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': -0.8, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 950, 'mismatch_pattern': 'Struggling Accurately'},
    {'obj_score': 0.4, 'conf_score': 0.4, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': -0.6, 'purpose': 'revision', 'prof_goal': 'basic', 'session': 'short', 'time_s': 900, 'mismatch_pattern': 'Struggling Accurately'},
    
    # G2 - Slow but Steady
    {'obj_score': 0.2, 'conf_score': 0.3, 'redo_count': 2, 'flagged_count': 2, 'conf_trend': 0.1, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 850, 'mismatch_pattern': 'Slow but Steady'},
    {'obj_score': 0.35, 'conf_score': 0.35, 'redo_count': 1, 'flagged_count': 1, 'conf_trend': -0.2, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 800, 'mismatch_pattern': 'Slow but Steady'},
    
    # H1 - Speed vs Accuracy Trade-off
    {'obj_score': 0.9, 'conf_score': 0.85, 'redo_count': 4, 'flagged_count': 1, 'conf_trend': -0.3, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 400, 'mismatch_pattern': 'Speed vs Accuracy Trade-off'},
    {'obj_score': 0.95, 'conf_score': 0.9, 'redo_count': 5, 'flagged_count': 0, 'conf_trend': 0.2, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 450, 'mismatch_pattern': 'Speed vs Accuracy Trade-off'},
    
    # H2 - Conceptual Blind Spots
    {'obj_score': 0.8, 'conf_score': 0.6, 'redo_count': 1, 'flagged_count': 4, 'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 600, 'mismatch_pattern': 'Conceptual Blind Spots'},
    {'obj_score': 0.85, 'conf_score': 0.65, 'redo_count': 0, 'flagged_count': 5, 'conf_trend': 0.8, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 550, 'mismatch_pattern': 'Conceptual Blind Spots'},
    
    # I1 - Slow Starter
    {'obj_score': 0.2, 'conf_score': 0.6, 'redo_count': 1, 'flagged_count': 0, 'conf_trend': 0.4, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 650, 'mismatch_pattern': 'Slow Starter'},
    {'obj_score': 0.35, 'conf_score': 0.65, 'redo_count': 0, 'flagged_count': 1, 'conf_trend': 0.7, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 600, 'mismatch_pattern': 'Slow Starter'},
    
    # I2 - Overestimating Understanding
    {'obj_score': 0.6, 'conf_score': 0.8, 'redo_count': 1, 'flagged_count': 4, 'conf_trend': -0.6, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 450, 'mismatch_pattern': 'Overestimating Understanding'},
    {'obj_score': 0.65, 'conf_score': 0.85, 'redo_count': 0, 'flagged_count': 5, 'conf_trend': -0.8, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 400, 'mismatch_pattern': 'Overestimating Understanding'},
    
    # J1a - Severely Overwhelmed
    {'obj_score': 0.15, 'conf_score': 0.15, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': -0.8, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 950, 'mismatch_pattern': 'Severely Overwhelmed'},
    {'obj_score': 0.25, 'conf_score': 0.2, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': -0.9, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 1000, 'mismatch_pattern': 'Severely Overwhelmed'},
    
    # J1b - Moderately Overwhelmed
    {'obj_score': 0.45, 'conf_score': 0.25, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': -0.6, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'short', 'time_s': 850, 'mismatch_pattern': 'Moderately Overwhelmed'},
    {'obj_score': 0.5, 'conf_score': 0.3, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': -0.7, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 900, 'mismatch_pattern': 'Moderately Overwhelmed'},
    
    # J1c - Content Overwhelmed
    {'obj_score': 0.65, 'conf_score': 0.35, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': -0.4, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 800, 'mismatch_pattern': 'Content Overwhelmed'},
    {'obj_score': 0.7, 'conf_score': 0.4, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': -0.5, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 750, 'mismatch_pattern': 'Content Overwhelmed'},
    
    # J2 - Experienced but Rusty
    {'obj_score': 0.8, 'conf_score': 0.5, 'redo_count': 1, 'flagged_count': 0, 'conf_trend': -0.8, 'purpose': 'revision', 'prof_goal': 'basic', 'session': 'short', 'time_s': 450, 'mismatch_pattern': 'Experienced but Rusty'},
    {'obj_score': 0.85, 'conf_score': 0.6, 'redo_count': 0, 'flagged_count': 1, 'conf_trend': -0.6, 'purpose': 'revision', 'prof_goal': 'basic', 'session': 'short', 'time_s': 400, 'mismatch_pattern': 'Experienced but Rusty'},
    
    # K1 - Delusional Beginner
    {'obj_score': 0.1, 'conf_score': 0.8, 'redo_count': 5, 'flagged_count': 5, 'conf_trend': 0.3, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 1200, 'mismatch_pattern': 'Delusional Beginner'},
    {'obj_score': 0.15, 'conf_score': 0.85, 'redo_count': 4, 'flagged_count': 4, 'conf_trend': 0.4, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 1100, 'mismatch_pattern': 'Delusional Beginner'},
    
    # K2 - Severe Impostor
    {'obj_score': 0.9, 'conf_score': 0.2, 'redo_count': 0, 'flagged_count': 0, 'conf_trend': -0.7, 'purpose': 'revision', 'prof_goal': 'basic', 'session': 'long', 'time_s': 950, 'mismatch_pattern': 'Severe Impostor'},
    {'obj_score': 0.95, 'conf_score': 0.25, 'redo_count': 0, 'flagged_count': 0, 'conf_trend': -0.8, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 1000, 'mismatch_pattern': 'Severe Impostor'},
    
    # L1 - Realistic Improver
    {'obj_score': 0.35, 'conf_score': 0.35, 'redo_count': 2, 'flagged_count': 2, 'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 700, 'mismatch_pattern': 'Realistic Improver'},
    {'obj_score': 0.4, 'conf_score': 0.4, 'redo_count': 3, 'flagged_count': 3, 'conf_trend': 0.6, 'purpose': 'scratch', 'prof_goal': 'basic', 'session': 'long', 'time_s': 750, 'mismatch_pattern': 'Realistic Improver'},
    
    # L2 - Balanced Learner
    {'obj_score': 0.7, 'conf_score': 0.7, 'redo_count': 2, 'flagged_count': 2, 'conf_trend': 0.2, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 600, 'mismatch_pattern': 'Balanced Learner'},
    {'obj_score': 0.75, 'conf_score': 0.75, 'redo_count': 1, 'flagged_count': 1, 'conf_trend': 0.3, 'purpose': 'revision', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 650, 'mismatch_pattern': 'Balanced Learner'},
    
    # O1 - All Topics Flagged - Surface Learning
    {'obj_score': 0.8, 'conf_score': 0.8, 'redo_count': 1, 'flagged_count': 5, 'conf_trend': 0.3, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'short', 'time_s': 550, 'mismatch_pattern': 'All Topics Flagged - Surface Learning'},
    {'obj_score': 0.85, 'conf_score': 0.85, 'redo_count': 2, 'flagged_count': 5, 'conf_trend': 0.7, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 500, 'mismatch_pattern': 'All Topics Flagged - Surface Learning'},
    
    # O2 - All Topics Flagged - Speed Reader
    {'obj_score': 0.9, 'conf_score': 0.9, 'redo_count': 0, 'flagged_count': 5, 'conf_trend': 0.8, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 400, 'mismatch_pattern': 'All Topics Flagged - Speed Reader'},
    {'obj_score': 0.95, 'conf_score': 0.95, 'redo_count': 1, 'flagged_count': 5, 'conf_trend': 0.9, 'purpose': 'revision', 'prof_goal': 'advanced', 'session': 'short', 'time_s': 450, 'mismatch_pattern': 'All Topics Flagged - Speed Reader'},
    
    # O3 - All Topics Flagged - False Mastery
    {'obj_score': 0.75, 'conf_score': 0.75, 'redo_count': 2, 'flagged_count': 5, 'conf_trend': 0.1, 'purpose': 'scratch', 'prof_goal': 'intermediate', 'session': 'long', 'time_s': 600, 'mismatch_pattern': 'All Topics Flagged - False Mastery'},
    {'obj_score': 0.8, 'conf_score': 0.8, 'redo_count': 3, 'flagged_count': 5, 'conf_trend': -0.2, 'purpose': 'scratch', 'prof_goal': 'advanced', 'session': 'long', 'time_s': 650, 'mismatch_pattern': 'All Topics Flagged - False Mastery'},
]

# === 2. Create DataFrame ===
df = pd.DataFrame(user_data)

# === 3. Define personalization strategies for each mismatch pattern ===
personalization_strategies = {
    'Perfect Alignment': "Advanced content, accelerated pace, challenge problems",
    'Consistent High': "Expert refresher, edge cases, applications",
    'Impostor Syndrome': "Confidence building, validation exercises, positive reinforcement",
    'Perfectionist Anxiety': "Stress management, 'good enough' messaging, time pressure reduction",
    'Overconfident Underperformer': "Reality check, diagnostic assessment, skill gap analysis",
    'Confident but Careless': "Attention to detail, careful review processes",
    'Dunning-Kruger Effect': "Foundational concepts, gentle reality adjustment",
    'False Confidence': "Comprehensive skill assessment, appropriate level placement",
    'Performance-Practice Disconnect': "Study habit analysis, learning method review",
    'Concept Confusion': "Targeted clarification, misconception addressing",
    'Analysis Paralysis': "Decision-making skills, simplified choices, clear guidance",
    'Practice Avoidance': "Motivation building, gamification, incremental challenges",
    'Struggling Accurately': "Intensive support, alternative methods, foundation building",
    'Slow but Steady': "Patient progression, encouragement, extended timelines",
    'Speed vs Accuracy Trade-off': "Time management, quality focus techniques",
    'Conceptual Blind Spots': "Targeted weak area focus, comprehensive review",
    'Slow Starter': "Patience, building momentum, progressive difficulty",
    'Overestimating Understanding': "Deeper comprehension checks, application exercises",
    'Severely Overwhelmed': "Micro-learning, frequent breaks, emotional support",
    'Moderately Overwhelmed': "Reduced cognitive load, structured guidance",
    'Content Overwhelmed': "Chunking strategies, prioritization",
    'Experienced but Rusty': "Quick refreshers, confidence restoration, gradual re-engagement",
    'Delusional Beginner': "Immediate intervention, reality alignment, foundation reset",
    'Severe Impostor': "Intensive confidence therapy, achievement recognition",
    'Realistic Improver': "Encouragement, skill building, progress tracking",
    'Balanced Learner': "Standard progression, balanced content",
    'All Topics Flagged - Surface Learning': "Deep comprehension checks, concept mastery verification",
    'All Topics Flagged - Speed Reader': "Slow down, quality over speed, thorough understanding",
    'All Topics Flagged - False Mastery': "Diagnostic deep-dive, misconception identification"
}

# === 4. Encode categorical variables ===
le_purpose = LabelEncoder().fit(df['purpose'])
le_prof_goal = LabelEncoder().fit(df['prof_goal'])
le_session = LabelEncoder().fit(df['session'])
le_mismatch = LabelEncoder().fit(df['mismatch_pattern'])

df['purpose_encoded'] = le_purpose.transform(df['purpose'])
df['prof_goal_encoded'] = le_prof_goal.transform(df['prof_goal'])
df['session_encoded'] = le_session.transform(df['session'])
df['mismatch_pattern_encoded'] = le_mismatch.transform(df['mismatch_pattern'])

# === 5. Prepare data for training ===
features = [
    'obj_score',
    'conf_score',
    'redo_count',
    'flagged_count',
    'conf_trend',
    'purpose_encoded',
    'prof_goal_encoded',
    'session_encoded',
    'time_s'
]
X = df[features]
y = df['mismatch_pattern_encoded']

# === 6. Train the decision tree model ===
model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X, y)

# === 7. Visualize using Graphviz ===
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=features,
    class_names=[html.escape(pattern) for pattern in le_mismatch.classes_],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=8  # Limit depth for readability
)

graph = graphviz.Source(dot_data)
graph.render("user_categorization_decision_tree", format="png", view=True)

# === 8. Save model and encoders ===
with open("user_categorization_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("user_categorization_encoders.pkl", "wb") as f:
    pickle.dump({
        'purpose_encoder': le_purpose,
        'prof_goal_encoder': le_prof_goal,
        'session_encoder': le_session,
        'mismatch_encoder': le_mismatch,
        'personalization_strategies': personalization_strategies,
        'features': features
    }, f)

print("User Categorization Model and encoders saved successfully!")
print("Decision tree visualization saved as 'user_categorization_decision_tree.png'")

# === 9. Display model performance ===
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on training data (for demonstration)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Number of training samples: {len(df)}")
print(f"Number of unique mismatch patterns: {len(le_mismatch.classes_)}")

# Display feature importance
feature_importance = model.feature_importances_
print("\nFeature Importance:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

# === 10. Test prediction function ===
def predict_user_category(obj_score, conf_score, redo_count, flagged_count, conf_trend, purpose, prof_goal, session, time_s):
    """
    Predict user categorization pattern based on input features
    """
    # Encode categorical inputs
    purpose_enc = le_purpose.transform([purpose])[0]
    prof_goal_enc = le_prof_goal.transform([prof_goal])[0]
    session_enc = le_session.transform([session])[0]
    
    # Create feature vector
    feature_vector = [[obj_score, conf_score, redo_count, flagged_count, conf_trend, 
                      purpose_enc, prof_goal_enc, session_enc, time_s]]
    
    # Make prediction
    prediction = model.predict(feature_vector)[0]
    mismatch_pattern = le_mismatch.inverse_transform([prediction])[0]
    strategy = personalization_strategies[mismatch_pattern]
    
    return mismatch_pattern, strategy

# Example prediction
example_pattern, example_strategy = predict_user_category(
    obj_score=0.3, conf_score=0.8, redo_count=4, flagged_count=4, 
    conf_trend=0.5, purpose='scratch', prof_goal='basic', session='short', time_s=600
)

print(f"\nExample Prediction:")
print(f"Input: obj_score=0.3, conf_score=0.8, redo_count=4, flagged_count=4, conf_trend=0.5")
print(f"       purpose='scratch', prof_goal='basic', session='short', time_s=600")
print(f"Predicted Pattern: {example_pattern}")
print(f"Strategy: {example_strategy}")
