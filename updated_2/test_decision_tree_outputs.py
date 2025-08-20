import json
import os
import warnings
import subprocess
from user_category.use_user_categorization_model import predict_user_category_with_skills
from redo_dec_tree.use_redo_decision_model import predict_redo_decision
from learner_intervention.use_learning_intervention_model import predict_learning_intervention
from topic_priority_manager import TopicPriorityManager

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Step 1: Preprocess sample_users.json using pre.py
subprocess.run(["python3", "pre.py"], check=True)

# Step 2: Load preprocessed user data from JSON file
with open("processed_users.json", "r") as f:
    processed_users = json.load(f)

results = []

# Initialize topic priority manager for tracking across modules
topic_manager = TopicPriorityManager(window_size=3)

for user in processed_users:
    # Use preprocessed features directly
    features = {
        "user_category": {
            "obj_score": user["avg_obj_score"],
            "conf_score": user["avg_conf_score"],
            "conf_trend": user["conf_trend"],
            "redo_count": user["redo_topics_count"],
            "flagged_count": user["flagged_topics_count"],
            "purpose": user["learner_purpose"],
            "prof_goal": user["desired_proficiency"],
            "session": user["session_preference"],
            "time_s": user["avg_time_module"],
            "skill_level": user["skill_level"]
        },
        "redo_decision": {
            "obj_score": user["avg_obj_score"],
            "redo_count": user["redo_topics_count"],
            "flagged_count": user["flagged_topics_count"]
        },
        "learning_intervention": {
            "redo_flag": user["redo_topics_count"] > 0,
            "session_preference": user["session_preference"],
            "time_taken": user["avg_time_module"]
        }
    }

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
print("- Critical topics (in both redo & flagged)")
print("- High priority topics (in either redo or flagged)")
print("- Weakness prompts generated based on current priority topics")
print(f"- Total modules processed: {len(processed_users)}")

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
