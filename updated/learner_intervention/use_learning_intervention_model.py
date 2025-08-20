import pickle
import numpy as np

def load_learning_intervention_model():
    """Load the trained learning intervention model and encoders"""
    try:
        # Load the trained model
        with open("learning_intervention_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load the encoders and metadata
        with open("learning_intervention_encoders.pkl", "rb") as f:
            encoders_data = pickle.load(f)
        
        return model, encoders_data
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run save_learning_intervention_model.py first.")
        print(f"Missing file: {e.filename}")
        return None, None

def validate_input(redo_flag, session_preference, time_taken):
    """Validate input parameters"""
    # Validate redo_flag
    if not isinstance(redo_flag, bool):
        raise ValueError("redo_flag must be a boolean (True/False)")
    
    # Validate session_preference
    if session_preference not in ['short', 'long']:
        raise ValueError("session_preference must be either 'short' or 'long'")
    
    # Validate time_taken (should be positive number)
    if not isinstance(time_taken, (int, float)) or time_taken <= 0:
        raise ValueError("time_taken must be a positive number (in seconds)")
    
    return True

def predict_learning_intervention(redo_flag, session_preference, time_taken, model=None, encoders_data=None):
    """
    Predict learning intervention based on redo flag, session preference, and time taken
    
    Args:
        redo_flag (bool): Whether the learner needs to redo content
        session_preference (str): 'short' or 'long' session preference
        time_taken (float): Time taken to complete module in seconds
        model: Pre-loaded model (optional)
        encoders_data: Pre-loaded encoders (optional)
    
    Returns:
        tuple: (intervention, confidence_score, description)
    """
    # Load model if not provided
    if model is None or encoders_data is None:
        model, encoders_data = load_learning_intervention_model()
        if model is None:
            return None, 0.0, ""
    
    # Validate input
    try:
        validate_input(redo_flag, session_preference, time_taken)
    except ValueError as e:
        print(f"Input validation error: {e}")
        return None, 0.0, ""
    
    # Get encoders
    session_encoder = encoders_data['session_encoder']
    intervention_encoder = encoders_data['intervention_encoder']
    intervention_descriptions = encoders_data['intervention_descriptions']
    features = encoders_data['features']
    
    # Prepare feature vector
    try:
        redo_encoded = int(redo_flag)
        session_encoded = session_encoder.transform([session_preference])[0]
        feature_vector = np.array([[redo_encoded, session_encoded, time_taken]])
        
        # Make prediction
        prediction_encoded = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Decode prediction
        intervention = intervention_encoder.inverse_transform([prediction_encoded])[0]
        confidence = probabilities[prediction_encoded]
        description = intervention_descriptions.get(intervention, "No description available")
        
        return intervention, confidence, description
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, ""

def get_time_category_explanation(time_taken, avg_time=720):
    """
    Provide explanation about time category relative to average
    """
    time_pct = (time_taken / avg_time) * 100
    
    if time_pct < 60:
        return f"Fast learner ({time_taken:.0f}s, {time_pct:.1f}% of avg)"
    elif time_pct <= 140:
        return f"Average pace ({time_taken:.0f}s, {time_pct:.1f}% of avg)"
    else:
        return f"Slow pace ({time_taken:.0f}s, {time_pct:.1f}% of avg)"

def get_intervention_explanation(redo_flag, session_preference, time_taken, intervention):
    """
    Provide detailed explanation for the intervention recommendation
    """
    explanations = []
    
    # Time analysis
    avg_time = 720  # Default average
    time_category = get_time_category_explanation(time_taken, avg_time)
    explanations.append(f"Time analysis: {time_category}")
    
    # Redo analysis
    if redo_flag:
        explanations.append("Redo flag: TRUE - Learner needs content repetition")
        
        # Specific redo scenarios
        if time_taken < 432:  # Fast but redoing
            explanations.append("Surface learning detected - completed quickly but needs repetition")
        elif time_taken <= 720:  # Average time but redoing
            explanations.append("Quick struggling - average time but comprehension issues")
        elif time_taken <= 1008:  # Slow but still manageable
            explanations.append("Deep struggling - taking longer and having comprehension issues")
        else:  # Very slow
            explanations.append("Critical struggling - significantly longer time with comprehension issues")
    else:
        explanations.append("Redo flag: FALSE - Learner progressing normally")
    
    # Session preference analysis
    explanations.append(f"Session preference: {session_preference.upper()}")
    
    # Override explanations for struggling learners
    if redo_flag and session_preference == 'short' and time_taken < 1008:
        explanations.append("Note: Short preference overridden to support deeper learning")
    
    return "; ".join(explanations)

def predict_batch_interventions(input_data):
    """
    Predict learning interventions for multiple cases
    
    Args:
        input_data (list): List of dictionaries with keys: redo_flag, session_preference, time_taken
    
    Returns:
        list: List of prediction results
    """
    # Load model once for batch processing
    model, encoders_data = load_learning_intervention_model()
    if model is None:
        return []
    
    results = []
    for i, data in enumerate(input_data):
        try:
            redo_flag = data['redo_flag']
            session_preference = data['session_preference']
            time_taken = data['time_taken']
            
            intervention, confidence, description = predict_learning_intervention(
                redo_flag, session_preference, time_taken, model, encoders_data
            )
            
            result = {
                'index': i,
                'redo_flag': redo_flag,
                'session_preference': session_preference,
                'time_taken': time_taken,
                'intervention': intervention,
                'confidence': confidence,
                'description': description,
                'explanation': get_intervention_explanation(redo_flag, session_preference, time_taken, intervention)
            }
            results.append(result)
            
        except KeyError as e:
            results.append({
                'index': i,
                'error': f"Missing key: {e}",
                'intervention': None,
                'confidence': 0.0,
                'description': ""
            })
        except Exception as e:
            results.append({
                'index': i,
                'error': f"Processing error: {e}",
                'intervention': None,
                'confidence': 0.0,
                'description': ""
            })
    
    return results

def get_intervention_recommendations_by_category():
    """
    Get intervention recommendations organized by learner categories
    """
    categories = {
        "Fast Learners (No Redo)": [
            {"redo_flag": False, "session_preference": "short", "time_taken": 350},
            {"redo_flag": False, "session_preference": "long", "time_taken": 400}
        ],
        "Average Learners (No Redo)": [
            {"redo_flag": False, "session_preference": "short", "time_taken": 600},
            {"redo_flag": False, "session_preference": "long", "time_taken": 800}
        ],
        "Slow Learners (No Redo)": [
            {"redo_flag": False, "session_preference": "short", "time_taken": 1200},
            {"redo_flag": False, "session_preference": "long", "time_taken": 1400}
        ],
        "Surface Learners (Fast + Redo)": [
            {"redo_flag": True, "session_preference": "short", "time_taken": 350},
            {"redo_flag": True, "session_preference": "long", "time_taken": 400}
        ],
        "Quick Struggling (Average + Redo)": [
            {"redo_flag": True, "session_preference": "short", "time_taken": 650},
            {"redo_flag": True, "session_preference": "long", "time_taken": 700}
        ],
        "Deep Struggling (Slow + Redo)": [
            {"redo_flag": True, "session_preference": "short", "time_taken": 900},
            {"redo_flag": True, "session_preference": "long", "time_taken": 950}
        ],
        "Critical Struggling (Very Slow + Redo)": [
            {"redo_flag": True, "session_preference": "short", "time_taken": 1500},
            {"redo_flag": True, "session_preference": "long", "time_taken": 1600}
        ]
    }
    
    print("Learning Intervention Recommendations by Category")
    print("=" * 60)
    
    for category, test_cases in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        results = predict_batch_interventions(test_cases)
        for result in results:
            if 'error' not in result:
                pref = result['session_preference']
                time = result['time_taken']
                intervention = result['intervention']
                confidence = result['confidence']
                print(f"  {pref.capitalize()} sessions ({time}s): {intervention} ({confidence:.3f})")
            else:
                print(f"  Error: {result['error']}")

# === Example usage ===
if __name__ == "__main__":
    print("Learning Intervention Prediction System")
    print("=" * 50)
    
    # Test cases covering different scenarios
    test_scenarios = [
        # Fast learners (no issues)
        {'redo_flag': False, 'session_preference': 'short', 'time_taken': 350, 'scenario': 'Fast Short'},
        {'redo_flag': False, 'session_preference': 'long', 'time_taken': 400, 'scenario': 'Fast Long'},
        
        # Average learners
        {'redo_flag': False, 'session_preference': 'short', 'time_taken': 600, 'scenario': 'Average Short'},
        {'redo_flag': False, 'session_preference': 'long', 'time_taken': 800, 'scenario': 'Average Long'},
        
        # Slow learners
        {'redo_flag': False, 'session_preference': 'short', 'time_taken': 1200, 'scenario': 'Slow Short'},
        {'redo_flag': False, 'session_preference': 'long', 'time_taken': 1400, 'scenario': 'Slow Long'},
        
        # Surface learners (fast but need redo)
        {'redo_flag': True, 'session_preference': 'short', 'time_taken': 350, 'scenario': 'Surface Short'},
        {'redo_flag': True, 'session_preference': 'long', 'time_taken': 400, 'scenario': 'Surface Long'},
        
        # Struggling learners
        {'redo_flag': True, 'session_preference': 'short', 'time_taken': 650, 'scenario': 'Quick Struggle Short'},
        {'redo_flag': True, 'session_preference': 'long', 'time_taken': 700, 'scenario': 'Quick Struggle Long'},
        
        # Critical cases
        {'redo_flag': True, 'session_preference': 'short', 'time_taken': 1500, 'scenario': 'Critical Short'},
        {'redo_flag': True, 'session_preference': 'long', 'time_taken': 1600, 'scenario': 'Critical Long'},
    ]
    
    # Run predictions
    for i, test_case in enumerate(test_scenarios, 1):
        redo_flag = test_case['redo_flag']
        session_pref = test_case['session_preference']
        time_taken = test_case['time_taken']
        scenario = test_case['scenario']
        
        intervention, confidence, description = predict_learning_intervention(redo_flag, session_pref, time_taken)
        
        print(f"\nTest {i:2d} - {scenario}:")
        print(f"  Input: redo={redo_flag}, session={session_pref}, time={time_taken}s")
        
        if intervention:
            print(f"  Intervention: {intervention}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Description: {description}")
            explanation = get_intervention_explanation(redo_flag, session_pref, time_taken, intervention)
            print(f"  Explanation: {explanation}")
        else:
            print("  Failed to make prediction")
    
    # Show category recommendations
    print(f"\n{'='*60}")
    get_intervention_recommendations_by_category()
    
    # Interactive mode
    print(f"\n{'='*50}")
    print("Interactive Mode")
    print("Enter 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nEnter redo_flag(true/false), session_preference(short/long), time_taken(seconds) or 'quit': ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse input
            values = [x.strip() for x in user_input.split(',')]
            if len(values) != 3:
                print("Please enter exactly 3 values separated by commas")
                continue
            
            redo_flag = values[0].lower() in ['true', '1', 'yes', 't']
            session_preference = values[1].lower()
            time_taken = float(values[2])
            
            if session_preference not in ['short', 'long']:
                print("Session preference must be 'short' or 'long'")
                continue
            
            # Make prediction
            intervention, confidence, description = predict_learning_intervention(redo_flag, session_preference, time_taken)
            
            if intervention:
                print(f"Intervention: {intervention}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Description: {description}")
                explanation = get_intervention_explanation(redo_flag, session_preference, time_taken, intervention)
                print(f"Explanation: {explanation}")
            else:
                print("Failed to make prediction. Please check your input.")
                
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")
