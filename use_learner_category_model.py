import pickle
import pandas as pd

# === Load the saved learner category model and encoders ===
with open("learner_category_model.pkl", "rb") as f:
    learner_category_model = pickle.load(f)

with open("learner_category_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
    le_level = encoders['level_encoder']
    le_purpose = encoders['purpose_encoder']
    le_category = encoders['category_encoder']
    category_prompts = encoders['category_prompts']
    features = encoders['features']

# === Function to calculate redo flag ===
def calculate_redo_flag(row):
    score = row['avg_objective_score']
    existing_redo = row['redo_topics']
    
    # Handle both list and integer formats for backward compatibility
    if isinstance(existing_redo, list):
        redo_count = len(existing_redo)
    else:
        redo_count = existing_redo
    
    if score < 5.5:  # Struggling
        return True
    elif 5.5 <= score < 6.5:  # Developing
        return redo_count >= 2
    elif 6.5 <= score < 8.0:  # Competent
        return redo_count >= 3
    else:  # Expert
        return False

# === Prediction function ===
def predict_learner_category(new_data, next_topic="the current topic"):
    """
    Predict learner category and generate appropriate prompt
    
    Args:
        new_data (dict): Dictionary containing learner data with keys:
            - avg_objective_score (float)
            - avg_confidence_score (float)
            - confidence_trend (float)
            - avg_skill_score (float)
            - learner_level (str): 'basic', 'intermediate', 'advanced'
            - learner_purpose (str): 'scratch', 'exploratory', 'revising'
            - flagged_topics (list or int): List of flagged topics or count
            - redo_topics (list or int): List of redo topics or count
            - stddev_objective_score (float)
        next_topic (str): The topic to generate prompt for
    
    Returns:
        dict: Contains category, redo_topics_flag, prompt, and confidence
    """
    # Convert lists to counts for model compatibility
    processed_data = new_data.copy()
    
    # Handle flagged_topics - convert list to count
    if isinstance(processed_data.get('flagged_topics'), list):
        processed_data['flagged_topics'] = len(processed_data['flagged_topics'])
    
    # Handle redo_topics - convert list to count  
    if isinstance(processed_data.get('redo_topics'), list):
        processed_data['redo_topics'] = len(processed_data['redo_topics'])
    
    # Ensure all values are scalar (not lists)
    for key, value in processed_data.items():
        if isinstance(value, list):
            # If any other field is a list, convert to length or first element
            processed_data[key] = len(value) if value else 0
    
    # Create DataFrame with a single row
    input_df = pd.DataFrame([processed_data])
    
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
    predicted_category_encoded = learner_category_model.predict(X_input)[0]
    predicted_category = le_category.inverse_transform([predicted_category_encoded])[0]
    
    # Generate prompt
    base_prompt = category_prompts[predicted_category]
    final_prompt = base_prompt.format(next_topic=next_topic)
    
    # Add redo context if needed
    redo_flag = input_df['redo_topics_flag'].iloc[0]
    if redo_flag:
        # Handle both list and integer formats
        redo_topics_data = new_data.get('redo_topics', 0)
        if isinstance(redo_topics_data, list) and len(redo_topics_data) > 0:
            final_prompt += f" Focus on reviewing and reinforcing previous topics that need attention: {', '.join(redo_topics_data)}."
        elif isinstance(redo_topics_data, int) and redo_topics_data > 0:
            final_prompt += f" Focus on reviewing and reinforcing previous topics that need attention."
    
    return {
        'category': predicted_category,
        'redo_topics_flag': redo_flag,
        'prompt': final_prompt,
        'confidence': max(learner_category_model.predict_proba(X_input)[0])
    }

# === Example usage ===
if __name__ == "__main__":
    # Example learner data
    new_learner = {
        'avg_objective_score': 6.5, 
        'avg_confidence_score': 0.55, 
        'confidence_trend': 0.01, 
        'avg_skill_score': 5.0, 
        'learner_level': 'intermediate', 
        'learner_purpose': 'revising', 
        'flagged_topics': 1, 
        'redo_topics': 0, 
        'stddev_objective_score': 0.4
    }
    
    # Get prediction
    result = predict_learner_category(new_learner, "machine learning fundamentals")
    print(f"Predicted Category: {result['category']}")
    print(f"Redo Topics Flag: {result['redo_topics_flag']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Generated Prompt: {result['prompt']}")