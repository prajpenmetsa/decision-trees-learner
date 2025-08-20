import json
import os
import warnings
from user_category.use_user_categorization_model import predict_user_category_with_skills
from redo_dec_tree.use_redo_decision_model import predict_redo_decision
from learner_intervention.use_learning_intervention_model import predict_learning_intervention
from topic_priority_manager import TopicPriorityManager

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 5 sample user inputs
sample_users = [
    {
        "module_id": "M1",
        "user_id": "U001",
        "objective_scores": [0.85, 0.9, 0.8],
        "confidence_scores": [0.8, 0.85, 0.9],##
        "confidence_previous": [0.7, 0.8, 0.85],
        "redo_topics": [],##
        "flagged_topics": [],
        "skill_level": {"logic": 0.8, "coding": 0.9, "memory": 0.85},
        "learner_purpose": "revision",
        "desired_proficiency": "advanced",
        "session_preference": "long",
        "time_module": [400, 420, 410]
    },
    {
        "module_id": "M2",
        "user_id": "U002",
        "objective_scores": [0.55, 0.6, 0.65],
        "confidence_scores": [0.6, 0.65, 0.7],
        "confidence_previous": [0.5, 0.6, 0.65],
        "redo_topics": ["topic1", "topic2"],
        "flagged_topics": ["topic2"],
        "skill_level": {"logic": 0.6, "coding": 0.65, "memory": 0.7},
        "learner_purpose": "scratch",
        "desired_proficiency": "intermediate",
        "session_preference": "short",
        "time_module": [600, 650, 700]
    },
    {
        "module_id": "M3",
        "user_id": "U003",
        "objective_scores": [0.3, 0.35, 0.4],
        "confidence_scores": [0.4, 0.45, 0.5],
        "confidence_previous": [0.5, 0.45, 0.4],
        "redo_topics": ["topic3", "topic4", "topic5"],
        "flagged_topics": ["topic4", "topic5"],
        "skill_level": {"logic": 0.4, "coding": 0.5, "memory": 0.45},
        "learner_purpose": "scratch",
        "desired_proficiency": "basic",
        "session_preference": "short",
        "time_module": [1200, 1300, 1250]
    },
    {
        "module_id": "M4",
        "user_id": "U004",
        "objective_scores": [0.75, 0.8, 0.78],
        "confidence_scores": [0.5, 0.55, 0.6],
        "confidence_previous": [0.6, 0.55, 0.5],
        "redo_topics": ["topic6"],
        "flagged_topics": ["topic6", "topic7"],
        "skill_level": {"logic": 0.7, "coding": 0.75, "memory": 0.8},
        "learner_purpose": "revision",
        "desired_proficiency": "intermediate",
        "session_preference": "long",
        "time_module": [800, 850, 900]
    },
    {
        "module_id": "M5",
        "user_id": "U005",
        "objective_scores": [0.45, 0.5, 0.55],
        "confidence_scores": [0.3, 0.35, 0.4],
        "confidence_previous": [0.4, 0.35, 0.3],
        "redo_topics": ["topic8", "topic9", "topic10", "topic11"],
        "flagged_topics": ["topic9", "topic10", "topic11"],
        "skill_level": {"logic": 0.5, "coding": 0.55, "memory": 0.6},
        "learner_purpose": "scratch",
        "desired_proficiency": "basic",
        "session_preference": "short",
        "time_module": [1000, 1050, 1100]
    }
]

def extract_features(user):
    # Preprocessing for decision trees
    avg_obj_score = sum(user["objective_scores"]) / len(user["objective_scores"])
    avg_conf_score = sum(user["confidence_scores"]) / len(user["confidence_scores"])
    conf_trend = (user["confidence_previous"][-1] - user["confidence_previous"][0]) / len(user["confidence_previous"])
    redo_count = len(user["redo_topics"])
    flagged_count = len(user["flagged_topics"])
    avg_time_module = sum(user["time_module"]) / len(user["time_module"])
    session_pref = user["session_preference"]
    # For learning intervention tree
    redo_flag = redo_count > 0
    return {
        "user_category": {
            "obj_score": avg_obj_score,
            "conf_score": avg_conf_score,
            "conf_trend": conf_trend,
            "redo_count": redo_count,
            "flagged_count": flagged_count,
            "purpose": user["learner_purpose"],
            "prof_goal": user["desired_proficiency"],
            "session": session_pref,
            "time_s": avg_time_module,
            "skill_level": user["skill_level"]
        },
        "redo_decision": {
            "obj_score": avg_obj_score,
            "redo_count": redo_count,
            "flagged_count": flagged_count
        },
        "learning_intervention": {
            "redo_flag": redo_flag,
            "session_preference": session_pref,
            "time_taken": avg_time_module
        }
    }

results = []

# Initialize topic priority manager for tracking across modules
topic_manager = TopicPriorityManager(window_size=3)

for user in sample_users:
    features = extract_features(user)
    
    # Update topic priorities for this user/module
    topic_manager.update_topic_tracker(
        user["module_id"],
        user["redo_topics"],
        user["flagged_topics"]
    )
    
    # Get topic priority information
    topic_summary = topic_manager.get_topic_summary()
    
    # User Category with Skill Analysis
    os.chdir(os.path.join(os.path.dirname(__file__), "user_category"))
    user_cat_result = predict_user_category_with_skills(**features["user_category"])
    os.chdir(os.path.join(os.path.dirname(__file__), "../redo_dec_tree".replace("../", "")))
    os.chdir(os.path.join(os.path.dirname(__file__), "redo_dec_tree"))
    redo_pred, redo_conf = predict_redo_decision(**features["redo_decision"])
    os.chdir(os.path.join(os.path.dirname(__file__), "learner_intervention"))
    intervention_pred, intervention_conf, intervention_desc = predict_learning_intervention(**features["learning_intervention"])
    os.chdir(os.path.dirname(__file__))
    results.append({
        "user_id": user["user_id"],
        "module_id": user["module_id"],
        "user_category": {
            "mismatch_pattern": user_cat_result['mismatch_pattern'],
            "personalization_strategy": user_cat_result['personalization_strategy'],
            "confidence": user_cat_result['confidence'],
            "skill_analysis": user_cat_result['skill_analysis']
        },
        "redo_decision": {"decision": redo_pred, "confidence": redo_conf},
        "learning_intervention": {"intervention": intervention_pred, "confidence": intervention_conf, "description": intervention_desc},
        "topic_priorities": {
            "current_priority_topics": topic_summary['current_priority_topics'],
            "weakness_prompts": topic_summary['weakness_prompts'],
            "topic_details": {
                "critical_topics": topic_summary['current_priority_topics']['critical'],
                "high_priority_topics": topic_summary['current_priority_topics']['high'],
                "total_tracked": topic_summary['total_tracked_topics']
            }
        }
    })

with open("test_decision_tree_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

print("Test outputs for 5 users with topic priority tracking saved to test_decision_tree_outputs.json")
print("\nTopic Priority Summary:")
print("- Critical topics (in both redo & flagged): Tracked for 3 modules")
print("- High priority topics (in either redo or flagged): Tracked for 1 module")
print("- Weakness prompts generated based on current priority topics")
print(f"- Total modules processed: {len(sample_users)}")

# Show final topic manager state
final_summary = topic_manager.get_topic_summary()
if final_summary['current_priority_topics']['critical'] or final_summary['current_priority_topics']['high']:
    print(f"\nFinal Active Topics:")
    if final_summary['current_priority_topics']['critical']:
        print(f"  Critical: {final_summary['current_priority_topics']['critical']}")
    if final_summary['current_priority_topics']['high']:
        print(f"  High Priority: {final_summary['current_priority_topics']['high']}")
else:
    print(f"\nNo topics currently being tracked (all have expired from sliding window)")

print(f"\nWeakness prompts available: {len(final_summary['weakness_prompts'])} categories")
