import pickle
import pandas as pd

# === Load the saved topic revisit model and encoders ===
with open("topic_revisit_model.pkl", "rb") as f:
    topic_revisit_model = pickle.load(f)

with open("topic_revisit_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
    features = encoders['features']
    target = encoders['target']

# === Prediction function ===
def predict_topic_revisit(avg_objective_score, redo_topics):
    """
    Predict if a learner needs to revisit topics based on avg_objective_score and redo_topics.

    Args:
        avg_objective_score (float): Average objective score.
        redo_topics (int): Number of redo topics.

    Returns:
        bool: True if the learner needs to revisit topics, False otherwise.
    """
    input_data = pd.DataFrame([{
        "avg_objective_score": avg_objective_score, 
        "redo_topics": redo_topics
    }])
    return topic_revisit_model.predict(input_data)[0]

def predict_topic_revisit_with_confidence(avg_objective_score, redo_topics):
    """
    Predict if a learner needs to revisit topics with confidence score.

    Args:
        avg_objective_score (float): Average objective score.
        redo_topics (int): Number of redo topics.

    Returns:
        dict: Contains revisit_needed (bool) and confidence (float).
    """
    input_data = pd.DataFrame([{
        "avg_objective_score": avg_objective_score, 
        "redo_topics": redo_topics
    }])
    
    prediction = topic_revisit_model.predict(input_data)[0]
    confidence = max(topic_revisit_model.predict_proba(input_data)[0])
    
    return {
        'revisit_needed': bool(prediction),
        'confidence': confidence
    }

# === Example usage ===
if __name__ == "__main__":
    # Example 1: High performing learner with few redo topics
    avg_objective_score = 8.0
    redo_topics = 1
    needs_revisit = predict_topic_revisit(avg_objective_score, redo_topics)
    print(f"Learner with score {avg_objective_score} and {redo_topics} redo topics:")
    print(f"Needs to revisit topics: {'Yes' if needs_revisit else 'No'}")
    
    # Example 2: Struggling learner with many redo topics
    avg_objective_score = 5.0
    redo_topics = 4
    result = predict_topic_revisit_with_confidence(avg_objective_score, redo_topics)
    print(f"\nLearner with score {avg_objective_score} and {redo_topics} redo topics:")
    print(f"Needs to revisit topics: {'Yes' if result['revisit_needed'] else 'No'}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Example 3: Intermediate learner
    avg_objective_score = 6.5
    redo_topics = 2
    result = predict_topic_revisit_with_confidence(avg_objective_score, redo_topics)
    print(f"\nLearner with score {avg_objective_score} and {redo_topics} redo topics:")
    print(f"Needs to revisit topics: {'Yes' if result['revisit_needed'] else 'No'}")
    print(f"Confidence: {result['confidence']:.2f}")