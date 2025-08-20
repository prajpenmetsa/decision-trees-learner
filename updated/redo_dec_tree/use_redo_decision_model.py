import pickle
import numpy as np

def load_redo_decision_model():
    """Load the trained redo decision model and encoders"""
    try:
        # Load the trained model
        with open("redo_decision_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load the encoders and metadata
        with open("redo_decision_encoders.pkl", "rb") as f:
            encoders_data = pickle.load(f)
        
        return model, encoders_data
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run save_redo_decision_model.py first.")
        print(f"Missing file: {e.filename}")
        return None, None

def validate_input(obj_score, redo_count, flagged_count):
    """Validate input parameters"""
    # Validate objective score
    if not (0 <= obj_score <= 1):
        raise ValueError("obj_score must be between 0 and 1")
    
    # Validate counts (should be non-negative integers)
    if not isinstance(redo_count, int) or redo_count < 0:
        raise ValueError("redo_count must be a non-negative integer")
    
    if not isinstance(flagged_count, int) or flagged_count < 0:
        raise ValueError("flagged_count must be a non-negative integer")
    
    return True

def predict_redo_decision(obj_score, redo_count, flagged_count, model=None, encoders_data=None):
    """
    Predict redo decision based on objective score, redo count, and flagged count
    
    Args:
        obj_score (float): Objective score between 0 and 1
        redo_count (int): Number of redo attempts
        flagged_count (int): Number of flagged items
        model: Pre-loaded model (optional)
        encoders_data: Pre-loaded encoders (optional)
    
    Returns:
        tuple: (prediction, confidence_score)
    """
    # Load model if not provided
    if model is None or encoders_data is None:
        model, encoders_data = load_redo_decision_model()
        if model is None:
            return None, 0.0
    
    # Validate input
    try:
        validate_input(obj_score, redo_count, flagged_count)
    except ValueError as e:
        print(f"Input validation error: {e}")
        return None, 0.0
    
    # Prepare feature vector
    features = encoders_data['features']
    feature_vector = np.array([[obj_score, redo_count, flagged_count]])
    
    # Make prediction
    try:
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Get confidence score for the prediction
        class_names = model.classes_
        if prediction == 'YES':
            confidence = probabilities[np.where(class_names == 'YES')[0][0]]
        else:
            confidence = probabilities[np.where(class_names == 'NO')[0][0]]
        
        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def get_decision_explanation(obj_score, redo_count, flagged_count, prediction):
    """
    Provide explanation for the redo decision based on the thresholds
    """
    explanations = []
    
    # Determine the score range
    if obj_score <= 0.4:
        explanations.append(f"Objective score {obj_score:.2f} is in range 0.0-0.4: Always recommend redo")
        return f"Decision: {prediction} - " + "; ".join(explanations)
    
    elif 0.4 < obj_score <= 0.6:
        range_name = "0.4-0.6"
        condition_met = (redo_count >= 2) or (flagged_count >= 2)
        explanations.append(f"Objective score {obj_score:.2f} is in range {range_name}")
        explanations.append(f"Condition: (Redo Count ≥ 2) OR (Flagged Count ≥ 2)")
        explanations.append(f"Current: Redo Count = {redo_count}, Flagged Count = {flagged_count}")
        explanations.append(f"Condition met: {condition_met}")
    
    elif 0.6 < obj_score <= 0.7:
        range_name = "0.6-0.7"
        condition_met = (redo_count >= 3) or (flagged_count >= 3)
        explanations.append(f"Objective score {obj_score:.2f} is in range {range_name}")
        explanations.append(f"Condition: (Redo Count ≥ 3) OR (Flagged Count ≥ 3)")
        explanations.append(f"Current: Redo Count = {redo_count}, Flagged Count = {flagged_count}")
        explanations.append(f"Condition met: {condition_met}")
    
    elif 0.7 < obj_score <= 0.8:
        range_name = "0.7-0.8"
        condition_met = (redo_count >= 4) or (flagged_count >= 4) or (redo_count >= 3 and flagged_count >= 2)
        explanations.append(f"Objective score {obj_score:.2f} is in range {range_name}")
        explanations.append(f"Condition: (Redo Count ≥ 4) OR (Flagged Count ≥ 4) OR ((Redo Count ≥ 3) AND (Flagged Count ≥ 2))")
        explanations.append(f"Current: Redo Count = {redo_count}, Flagged Count = {flagged_count}")
        explanations.append(f"Condition met: {condition_met}")
    
    else:  # 0.8 < obj_score <= 1.0
        range_name = "0.8-1.0"
        condition_met = (redo_count >= 5) or (flagged_count >= 5) or \
                       (redo_count >= 4 and flagged_count >= 2) or \
                       (redo_count >= 3 and flagged_count >= 3)
        explanations.append(f"Objective score {obj_score:.2f} is in range {range_name}")
        explanations.append(f"Condition: (Redo Count ≥ 5) OR (Flagged Count ≥ 5) OR ((Redo Count ≥ 4) AND (Flagged Count ≥ 2)) OR ((Redo Count ≥ 3) AND (Flagged Count ≥ 3))")
        explanations.append(f"Current: Redo Count = {redo_count}, Flagged Count = {flagged_count}")
        explanations.append(f"Condition met: {condition_met}")
    
    return f"Decision: {prediction} - " + "; ".join(explanations)

def predict_batch(input_data):
    """
    Predict redo decisions for multiple cases
    
    Args:
        input_data (list): List of dictionaries with keys: obj_score, redo_count, flagged_count
    
    Returns:
        list: List of prediction results
    """
    # Load model once for batch processing
    model, encoders_data = load_redo_decision_model()
    if model is None:
        return []
    
    results = []
    for i, data in enumerate(input_data):
        try:
            obj_score = data['obj_score']
            redo_count = data['redo_count']
            flagged_count = data['flagged_count']
            
            prediction, confidence = predict_redo_decision(
                obj_score, redo_count, flagged_count, model, encoders_data
            )
            
            result = {
                'index': i,
                'obj_score': obj_score,
                'redo_count': redo_count,
                'flagged_count': flagged_count,
                'prediction': prediction,
                'confidence': confidence,
                'explanation': get_decision_explanation(obj_score, redo_count, flagged_count, prediction)
            }
            results.append(result)
            
        except KeyError as e:
            results.append({
                'index': i,
                'error': f"Missing key: {e}",
                'prediction': None,
                'confidence': 0.0
            })
        except Exception as e:
            results.append({
                'index': i,
                'error': f"Processing error: {e}",
                'prediction': None,
                'confidence': 0.0
            })
    
    return results

# === Example usage ===
if __name__ == "__main__":
    print("Redo Decision Prediction System")
    print("=" * 50)
    
    # Test cases covering different score ranges
    test_cases = [
        # 0.0-0.4 range (always YES)
        {'obj_score': 0.3, 'redo_count': 0, 'flagged_count': 0},
        {'obj_score': 0.2, 'redo_count': 1, 'flagged_count': 1},
        
        # 0.4-0.6 range
        {'obj_score': 0.5, 'redo_count': 0, 'flagged_count': 1},  # Should be NO
        {'obj_score': 0.5, 'redo_count': 2, 'flagged_count': 0},  # Should be YES
        {'obj_score': 0.5, 'redo_count': 0, 'flagged_count': 2},  # Should be YES
        
        # 0.6-0.7 range
        {'obj_score': 0.65, 'redo_count': 2, 'flagged_count': 2},  # Should be NO
        {'obj_score': 0.65, 'redo_count': 3, 'flagged_count': 0},  # Should be YES
        {'obj_score': 0.65, 'redo_count': 0, 'flagged_count': 3},  # Should be YES
        
        # 0.7-0.8 range
        {'obj_score': 0.75, 'redo_count': 3, 'flagged_count': 2},  # Should be YES
        {'obj_score': 0.75, 'redo_count': 3, 'flagged_count': 1},  # Should be NO
        {'obj_score': 0.75, 'redo_count': 4, 'flagged_count': 0},  # Should be YES
        
        # 0.8-1.0 range
        {'obj_score': 0.9, 'redo_count': 3, 'flagged_count': 3},   # Should be YES
        {'obj_score': 0.9, 'redo_count': 4, 'flagged_count': 2},   # Should be YES
        {'obj_score': 0.9, 'redo_count': 2, 'flagged_count': 2},   # Should be NO
    ]
    
    # Run batch prediction
    results = predict_batch(test_cases)
    
    for result in results:
        if 'error' in result:
            print(f"Test {result['index']+1}: Error - {result['error']}")
        else:
            print(f"\nTest {result['index']+1}:")
            print(f"  Input: obj_score={result['obj_score']:.2f}, redo_count={result['redo_count']}, flagged_count={result['flagged_count']}")
            print(f"  Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            print(f"  Explanation: {result['explanation']}")
    
    # Interactive mode
    print(f"\n{'='*50}")
