import pickle
import pandas as pd

# === Load the saved strategy model and encoders ===
with open("strategy_model.pkl", "rb") as f:
    strategy_model = pickle.load(f)

with open("strategy_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
    le_session = encoders['session_encoder']
    le_format = encoders['format_encoder']
    le_strategy = encoders['strategy_encoder']
    strategy_prompts = encoders['strategy_prompts']
    features = encoders['features']

# === Prediction function ===
def predict_strategy(revisiting_module, session_preference, attention_span, format_preference):
    """
    Predict learning strategy based on input parameters.
    
    Args:
        revisiting_module (bool): True if learner is revisiting, False if progressing
        session_preference (str): 'short_chunks' or 'long_sessions'
        attention_span (float): Attention span in minutes
        format_preference (str): 'text', 'image', or 'video'
    
    Returns:
        dict: Contains strategy_label, strategy_prompt, and confidence
    """
    # Create DataFrame with input data
    input_data = pd.DataFrame([{
        'revisiting_module_encoded': int(revisiting_module),
        'session_preference_encoded': le_session.transform([session_preference])[0],
        'attention_span': attention_span,
        'format_preference_encoded': le_format.transform([format_preference])[0]
    }])
    
    # Make prediction
    predicted_strategy_encoded = strategy_model.predict(input_data)[0]
    predicted_strategy = le_strategy.inverse_transform([predicted_strategy_encoded])[0]
    
    # Get strategy prompt
    strategy_prompt = strategy_prompts[predicted_strategy]
    
    # Get confidence
    confidence = max(strategy_model.predict_proba(input_data)[0])
    
    return {
        'strategy_label': predicted_strategy,
        'strategy_prompt': strategy_prompt,
        'confidence': confidence
    }

def get_user_category_description(revisiting_module, session_preference, attention_span, format_preference):
    """
    Generate a descriptive category for the user based on input parameters.
    """
    revisiting = "Revisiting" if revisiting_module else "Progressing"
    session_pref = session_preference.replace('_', ' ').title()
    format_pref = format_preference.title()

    if attention_span < 9:
        att_cat = "Low (<9 min)"
    elif attention_span >= 11:
        att_cat = "High (>=11 min)"
    else:
        att_cat = "Intermediate (9-11 min)"

    return f"{revisiting} | Session: {session_pref} | Attention: {att_cat} | Format: {format_pref}"

# === Example usage ===
if __name__ == "__main__":
    # Example 1: Revisiting learner with short attention span
    revisiting_module = True
    session_preference = 'short_chunks'
    attention_span = 7.5
    format_preference = 'text'
    
    result = predict_strategy(revisiting_module, session_preference, attention_span, format_preference)
    user_desc = get_user_category_description(revisiting_module, session_preference, attention_span, format_preference)
    
    print(f"User Category: {user_desc}")
    print(f"Predicted Strategy: {result['strategy_label']}")
    print(f"Strategy Prompt: {result['strategy_prompt']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Example 2: Progressing learner with long sessions
    print("\n" + "="*50)
    revisiting_module = False
    session_preference = 'long_sessions'
    attention_span = 12.0
    format_preference = 'image'
    
    result = predict_strategy(revisiting_module, session_preference, attention_span, format_preference)
    user_desc = get_user_category_description(revisiting_module, session_preference, attention_span, format_preference)
    
    print(f"User Category: {user_desc}")
    print(f"Predicted Strategy: {result['strategy_label']}")
    print(f"Strategy Prompt: {result['strategy_prompt']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Example 3: Intermediate attention span
    print("\n" + "="*50)
    revisiting_module = True
    session_preference = 'long_sessions'
    attention_span = 9.8
    format_preference = 'video'
    
    result = predict_strategy(revisiting_module, session_preference, attention_span, format_preference)
    user_desc = get_user_category_description(revisiting_module, session_preference, attention_span, format_preference)
    
    print(f"User Category: {user_desc}")
    print(f"Predicted Strategy: {result['strategy_label']}")
    print(f"Strategy Prompt: {result['strategy_prompt']}")
    print(f"Confidence: {result['confidence']:.2f}")