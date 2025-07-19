import pickle
import pandas as pd

# Load the saved model and encoders
with open("learner_decision_tree_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("learner_level_encoder.pkl", "rb") as level_encoder_file:
    le_level = pickle.load(level_encoder_file)

with open("learner_purpose_encoder.pkl", "rb") as purpose_encoder_file:
    le_purpose = pickle.load(purpose_encoder_file)

with open("learner_category_encoder.pkl", "rb") as category_encoder_file:
    le_category = pickle.load(category_encoder_file)

# Features used in the model
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

def predict_learner_category(new_data):
    """
    Predict the category of a learner based on input data.

    Args:
        new_data (dict): Dictionary containing learner data.

    Returns:
        str: Predicted learner category.
    """
    # Create DataFrame with a single row
    input_df = pd.DataFrame([new_data])

    # Encode categorical variables
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

# Example usage
if __name__ == "__main__":
    new_learner = {
        'avg_objective_score': 10,
        'avg_confidence_score': 0.5,
        'confidence_trend': 0.06,
        'avg_skill_score': 4.8,
        'learner_level': 'intermediate',
        'learner_purpose': 'revising',
        'flagged_topics': 1,
        'redo_topics': 0,
        'stddev_objective_score': 0.4
    }

    predicted_category = predict_learner_category(new_learner)
    print(f"Predicted learner category: {predicted_category}")