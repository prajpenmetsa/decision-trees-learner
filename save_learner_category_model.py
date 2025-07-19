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

# === 2. Create DataFrame ===
df = pd.DataFrame(learner_data)

# === 3. Define MUTUALLY EXCLUSIVE prompt mappings ===
category_prompts = {
    'Struggling Novice': "For {next_topic}: Use very simple language and break into tiny steps. Provide frequent encouragement and basic examples. Focus on building fundamental confidence with achievable milestones.",
    'Hesitant Learner': "For {next_topic}: Use gentle scaffolding with clear structure. Provide reassurance and validate small progress. Address uncertainty with supportive, patient feedback.",
    'Overconfident Novice': "For {next_topic}: Gently challenge assumptions with evidence-based examples. Use misconception-targeting approach while maintaining motivation. Provide reality checks with supportive correction.",
    'Anxious Improver': "For {next_topic}: Emphasize existing progress and capabilities. Use positive reinforcement to build confidence. Focus on success recognition and reduce self-doubt through structured guidance.",
    'Rising Improver': "For {next_topic}: Provide moderate challenges with clear progress indicators. Use supportive feedback to maintain momentum. Celebrate improvements while encouraging steady advancement.",
    'Overreacher': "For {next_topic}: Include appropriate challenges with reality checks. Provide just-in-time hints to prevent overextension. Guide ambitious goals with structured, realistic support.",
    'Confidence-Delayed': "For {next_topic}: Highlight existing competence and reduce hand-holding. Build independent confidence by emphasizing capability over support. Focus on self-trust development.",
    'Steady Performer': "For {next_topic}: Provide standard-paced content with moderate challenges. Include brief concept checks and maintain consistent progression. Focus on reliable skill building.",
    'Confident Achiever': "For {next_topic}: Deliver engaging content with varied challenges. Encourage exploration and creative problem-solving. Provide opportunities for practical application and innovation.",
    'Humble Expert': "For {next_topic}: Present advanced content with nuanced discussions. Acknowledge expertise while encouraging continued growth. Focus on mastery refinement and deeper understanding.",
    'Stable Expert': "For {next_topic}: Generate expert-level content with cutting-edge concepts. Minimal guidance needed - encourage innovation and advanced applications. Focus on thought leadership.",
    'Imposter Syndrome': "For {next_topic}: Affirm existing expertise while teaching. Provide clear evidence of competence to reduce self-doubt. Focus on recognizing and celebrating mastery already achieved."
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
    max_depth=10,
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
    class_names=[html.escape(label) for label in le_category.classes_],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=5
)

graph = graphviz.Source(dot_data)
graph.render("learner_category_decision_tree", format="png", view=True)

# === 8. Save model and encoders ===
with open("learner_category_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("learner_category_encoders.pkl", "wb") as f:
    pickle.dump({
        'level_encoder': le_level,
        'purpose_encoder': le_purpose,
        'category_encoder': le_category,
        'category_prompts': category_prompts,
        'features': features
    }, f)

print("Learner Category Model and encoders saved successfully!")
print("Decision tree visualization saved as 'learner_category_decision_tree.png'")