import pickle
import numpy as np

# === Skill-based content recommendations ===
SKILL_CONTENT_PROMPTS = {
    'logic': "Focus on problem-solving scenarios, decision trees, and algorithmic thinking exercises. Present content with clear logical flow and step-by-step reasoning.",
    'coding': "Emphasize hands-on programming exercises, code examples, and practical implementations. Provide interactive coding challenges and real-world project applications.",
    'memory': "Use structured information, visual aids, and repetitive reinforcement techniques. Organize content with clear hierarchies, summaries, and memory retention strategies."
}

# === Load the trained model and encoders ===
def load_user_categorization_model():
    """
    Load the trained user categorization model and encoders
    """
    with open("user_categorization_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("user_categorization_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    
    return model, encoders

# === Skill analysis functions ===
def analyze_user_skills(skill_level, threshold=0.7):
    """
    Analyze user skills and determine the strongest skill
    
    Args:
        skill_level (dict): Dictionary with skill scores {'logic': 0.4, 'coding': 0.6, 'memory': 0.5}
        threshold (float): Minimum threshold for skill mastery (default: 0.7)
    
    Returns:
        tuple: (strongest_skill, skill_score, skill_statement)
    """
    if not skill_level or not isinstance(skill_level, dict):
        return None, 0.0, "No skill data available"
    
    # Find the skill with the highest score
    strongest_skill = max(skill_level, key=skill_level.get)
    skill_score = skill_level[strongest_skill]
    
    # Generate simple skill statement
    if skill_score >= threshold:
        level = "master"
    elif skill_score >= 0.6:
        level = "proficient"
    elif skill_score >= 0.4:
        level = "good"
    else:
        level = "developing"
    
    skill_statement = f"User is {level} in {strongest_skill} ({skill_score:.2f}) - this is their strongest area"
    
    return strongest_skill, skill_score, skill_statement

def get_skill_based_content_recommendation(strongest_skill):
    """
    Get content recommendation based on user's strongest skill
    
    Args:
        strongest_skill (str): The user's strongest skill ('logic', 'coding', 'memory')
    
    Returns:
        str: Simple skill acknowledgment
    """
    if strongest_skill is None:
        return "No specific skill strength identified"
    
    return f"Focus on {strongest_skill}-based learning approaches"

# === Enhanced prediction function with skill analysis ===
def predict_user_category_with_skills(obj_score, conf_score, redo_count, flagged_count, conf_trend, 
                                    purpose, prof_goal, session, time_s, skill_level=None, 
                                    model=None, encoders=None):
    """
    Predict user categorization pattern and get personalization strategy with skill analysis
    
    Args:
        obj_score (float): Objective score (0.0 - 1.0)
        conf_score (float): Confidence score (0.0 - 1.0) 
        redo_count (int): Number of topics redone (0-5)
        flagged_count (int): Number of flagged topics (0-5)
        conf_trend (float): Confidence trend (-1.0 to 1.0)
        purpose (str): Learning purpose ('scratch' or 'revision')
        prof_goal (str): Proficiency goal ('basic', 'intermediate', 'advanced')
        session (str): Session preference ('short' or 'long')
        time_s (int): Time spent in seconds
        skill_level (dict): Skill scores {'logic': 0.4, 'coding': 0.6, 'memory': 0.5}
        model: Pre-loaded model (optional)
        encoders: Pre-loaded encoders (optional)
    
    Returns:
        dict: Complete user analysis with category, strategy, and skill recommendations
    """
    # Get basic category prediction
    mismatch_pattern, strategy, confidence = predict_user_category(
        obj_score, conf_score, redo_count, flagged_count, conf_trend,
        purpose, prof_goal, session, time_s, model, encoders
    )
    
    # Analyze skills
    strongest_skill, skill_score, skill_statement = analyze_user_skills(skill_level)
    
    # Create comprehensive result
    result = {
        'mismatch_pattern': mismatch_pattern,
        'personalization_strategy': strategy,
        'confidence': confidence,
        'skill_analysis': {
            'strongest_skill': strongest_skill,
            'skill_score': skill_score,
            'skill_statement': skill_statement
        }
    }
    
    return result
# === Original prediction function (maintained for backward compatibility) ===
def predict_user_category(obj_score, conf_score, redo_count, flagged_count, conf_trend, 
                         purpose, prof_goal, session, time_s, model=None, encoders=None):
    """
    Predict user categorization pattern and get personalization strategy
    
    Args:
        obj_score (float): Objective score (0.0 - 1.0)
        conf_score (float): Confidence score (0.0 - 1.0) 
        redo_count (int): Number of topics redone (0-5)
        flagged_count (int): Number of flagged topics (0-5)
        conf_trend (float): Confidence trend (-1.0 to 1.0)
        purpose (str): Learning purpose ('scratch' or 'revision')
        prof_goal (str): Proficiency goal ('basic', 'intermediate', 'advanced')
        session (str): Session preference ('short' or 'long')
        time_s (int): Time spent in seconds
        model: Pre-loaded model (optional)
        encoders: Pre-loaded encoders (optional)
    
    Returns:
        tuple: (mismatch_pattern, personalization_strategy, confidence)
    """
    # Load model and encoders if not provided
    if model is None or encoders is None:
        model, encoders = load_user_categorization_model()
    
    # Extract encoders
    purpose_encoder = encoders['purpose_encoder']
    prof_goal_encoder = encoders['prof_goal_encoder']
    session_encoder = encoders['session_encoder']
    mismatch_encoder = encoders['mismatch_encoder']
    strategies = encoders['personalization_strategies']
    
    # Validate inputs
    if purpose not in purpose_encoder.classes_:
        raise ValueError(f"Invalid purpose. Must be one of: {list(purpose_encoder.classes_)}")
    if prof_goal not in prof_goal_encoder.classes_:
        raise ValueError(f"Invalid prof_goal. Must be one of: {list(prof_goal_encoder.classes_)}")
    if session not in session_encoder.classes_:
        raise ValueError(f"Invalid session. Must be one of: {list(session_encoder.classes_)}")
    
    # Encode categorical inputs
    purpose_enc = purpose_encoder.transform([purpose])[0]
    prof_goal_enc = prof_goal_encoder.transform([prof_goal])[0]
    session_enc = session_encoder.transform([session])[0]
    
    # Create feature vector
    feature_vector = [[obj_score, conf_score, redo_count, flagged_count, conf_trend, 
                      purpose_enc, prof_goal_enc, session_enc, time_s]]
    
    # Make prediction
    prediction = model.predict(feature_vector)[0]
    prediction_proba = model.predict_proba(feature_vector)[0]
    
    # Get mismatch pattern and strategy
    mismatch_pattern = mismatch_encoder.inverse_transform([prediction])[0]
    strategy = strategies[mismatch_pattern]
    confidence = prediction_proba[prediction]
    
    return mismatch_pattern, strategy, confidence

# === Enhanced batch prediction function ===
def predict_multiple_users_with_skills(user_data_list, model=None, encoders=None):
    """
    Predict categories for multiple users with skill analysis
    
    Args:
        user_data_list: List of dictionaries with user data (including skill_level)
        model: Pre-loaded model (optional)
        encoders: Pre-loaded encoders (optional)
    
    Returns:
        List of dictionaries: [result_dict, ...]
    """
    if model is None or encoders is None:
        model, encoders = load_user_categorization_model()
    
    results = []
    for user_data in user_data_list:
        try:
            result = predict_user_category_with_skills(
                user_data['obj_score'], user_data['conf_score'], 
                user_data['redo_count'], user_data['flagged_count'],
                user_data['conf_trend'], user_data['purpose'], 
                user_data['prof_goal'], user_data['session'], 
                user_data['time_s'], user_data.get('skill_level', {}),
                model, encoders
            )
            results.append(result)
        except Exception as e:
            results.append({
                'error': str(e),
                'mismatch_pattern': None,
                'personalization_strategy': None,
                'confidence': 0.0,
                'skill_analysis': {
                    'strongest_skill': None,
                    'skill_score': 0.0,
                    'skill_statement': "Error in analysis"
                }
            })
    
    return results
def predict_multiple_users(user_data_list, model=None, encoders=None):
    """
    Predict categories for multiple users
    
    Args:
        user_data_list: List of dictionaries with user data
        model: Pre-loaded model (optional)
        encoders: Pre-loaded encoders (optional)
    
    Returns:
        List of tuples: [(mismatch_pattern, strategy, confidence), ...]
    """
    if model is None or encoders is None:
        model, encoders = load_user_categorization_model()
    
    results = []
    for user_data in user_data_list:
        try:
            pattern, strategy, confidence = predict_user_category(
                user_data['obj_score'], user_data['conf_score'], 
                user_data['redo_count'], user_data['flagged_count'],
                user_data['conf_trend'], user_data['purpose'], 
                user_data['prof_goal'], user_data['session'], 
                user_data['time_s'], model, encoders
            )
            results.append((pattern, strategy, confidence))
        except Exception as e:
            results.append((f"Error: {str(e)}", "", 0.0))
    
    return results

# === Example usage and testing with skill analysis ===
if __name__ == "__main__":
    print("User Categorization Model with Skill Analysis - Testing")
    print("=" * 60)
    
    # Load model once for multiple predictions
    model, encoders = load_user_categorization_model()
    
    # Test cases with skill data included
    test_cases = [
        {
            'name': 'High Performer - Logic Master',
            'data': {
                'obj_score': 0.9, 'conf_score': 0.85, 'redo_count': 0, 'flagged_count': 0,
                'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'advanced', 
                'session': 'long', 'time_s': 450,
                'skill_level': {'logic': 0.8, 'coding': 0.6, 'memory': 0.5}
            }
        },
        {
            'name': 'Impostor Syndrome - Coding Expert',
            'data': {
                'obj_score': 0.8, 'conf_score': 0.3, 'redo_count': 1, 'flagged_count': 2,
                'conf_trend': -0.5, 'purpose': 'scratch', 'prof_goal': 'basic', 
                'session': 'long', 'time_s': 800,
                'skill_level': {'logic': 0.5, 'coding': 0.75, 'memory': 0.4}
            }
        },
        {
            'name': 'Struggling Learner - Memory Proficient',
            'data': {
                'obj_score': 0.3, 'conf_score': 0.1, 'redo_count': 3, 'flagged_count': 4,
                'conf_trend': 0.5, 'purpose': 'scratch', 'prof_goal': 'basic', 
                'session': 'long', 'time_s': 600,
                'skill_level': {'logic': 0.3, 'coding': 0.4, 'memory': 0.72}
            }
        },
        {
            'name': 'Balanced Learner - No Mastery',
            'data': {
                'obj_score': 0.3, 'conf_score': 0.25, 'redo_count': 5, 'flagged_count': 5,
                'conf_trend': -0.8, 'purpose': 'scratch', 'prof_goal': 'basic', 
                'session': 'short', 'time_s': 950,
                'skill_level': {'logic': 0.5, 'coding': 0.6, 'memory': 0.55}
            }
        },
        {
            'name': 'All Topics Flagged - Multiple Skills Above Threshold',
            'data': {
                'obj_score': 0.9, 'conf_score': 0.9, 'redo_count': 0, 'flagged_count': 5,
                'conf_trend': 0.8, 'purpose': 'revision', 'prof_goal': 'advanced', 
                'session': 'short', 'time_s': 400,
                'skill_level': {'logic': 0.85, 'coding': 0.78, 'memory': 0.65}
            }
        }
    ]
    
    # Run enhanced predictions with skill analysis
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        data = test_case['data']
        print(f"Input: obj_score={data['obj_score']}, conf_score={data['conf_score']}")
        print(f"       redo_count={data['redo_count']}, flagged_count={data['flagged_count']}")
        print(f"       conf_trend={data['conf_trend']}, purpose='{data['purpose']}'")
        print(f"       prof_goal='{data['prof_goal']}', session='{data['session']}'")
        print(f"       time_s={data['time_s']}")
        print(f"       skills: {data['skill_level']}")
        
        try:
            result = predict_user_category_with_skills(
                **data, model=model, encoders=encoders
            )
            print(f"\nResults:")
            print(f"  Category: {result['mismatch_pattern']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Strategy: {result['personalization_strategy']}")
            print(f"  Skill Analysis: {result['skill_analysis']['skill_statement']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Test skill analysis separately
    print(f"\n{'='*60}")
    print("Skill Analysis Testing:")
    print("-" * 30)
    
    skill_test_cases = [
        {'logic': 0.8, 'coding': 0.6, 'memory': 0.5},  # Logic master
        {'logic': 0.5, 'coding': 0.75, 'memory': 0.4},  # Coding master
        {'logic': 0.3, 'coding': 0.4, 'memory': 0.72},  # Memory master
        {'logic': 0.6, 'coding': 0.65, 'memory': 0.55},  # No mastery, coding strongest
        {},  # No skill data
    ]
    
    for i, skills in enumerate(skill_test_cases, 1):
        strongest_skill, skill_score, skill_statement = analyze_user_skills(skills)
        print(f"Skills {i}: {skills}")
        print(f"  Analysis: {skill_statement}")
        print()
    
    print(f"\n{'='*60}")
    print("Feature Information:")
    print("-" * 20)
    print("Available values:")
    print(f"- purpose: {list(encoders['purpose_encoder'].classes_)}")
    print(f"- prof_goal: {list(encoders['prof_goal_encoder'].classes_)}")
    print(f"- session: {list(encoders['session_encoder'].classes_)}")
    print(f"- obj_score, conf_score: 0.0 - 1.0")
    print(f"- redo_count, flagged_count: 0 - 5")
    print(f"- conf_trend: -1.0 to 1.0")
    print(f"- time_s: positive integer (seconds)")
    print(f"- skill_level: dict with 'logic', 'coding', 'memory' keys (0.0 - 1.0)")
    print(f"- skill mastery threshold: 0.7")
