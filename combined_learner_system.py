import pickle
import pandas as pd
import json
import numpy as np
from use_learner_category_model import predict_learner_category
from use_topic_revisit_model import predict_topic_revisit_with_confidence
from use_strategy_model import predict_strategy, get_user_category_description

class CombinedLearnerSystem:
    """
    Combined system that integrates all three decision trees:
    1. Learner Category Model
    2. Topic Revisit Model  
    3. Strategy Model
    """
    
    def __init__(self):
        """Initialize the combined system by loading all models."""
        self.load_models()
    
    def load_models(self):
        """Load all three models and their encoders."""
        try:
            # Load learner category model
            with open("learner_category_model.pkl", "rb") as f:
                self.learner_category_model = pickle.load(f)
            with open("learner_category_encoders.pkl", "rb") as f:
                self.learner_category_encoders = pickle.load(f)
            
            # Load topic revisit model
            with open("topic_revisit_model.pkl", "rb") as f:
                self.topic_revisit_model = pickle.load(f)
            with open("topic_revisit_encoders.pkl", "rb") as f:
                self.topic_revisit_encoders = pickle.load(f)
            
            # Load strategy model
            with open("strategy_model.pkl", "rb") as f:
                self.strategy_model = pickle.load(f)
            with open("strategy_encoders.pkl", "rb") as f:
                self.strategy_encoders = pickle.load(f)
            
            print("All models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please ensure all model files exist by running the save_*_model.py scripts first.")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def predict_combined(self, learner_data, strategy_data, topic_name="machine learning"):
        """
        Make predictions using all three models and combine results.
        
        Args:
            learner_data (dict): Data for learner category prediction
            strategy_data (dict): Data for strategy prediction
            topic_name (str): The current topic name (only used when revisiting)
        
        Returns:
            dict: Combined results from all three models in JSON format
        """
        # 1. Extract topic lists and counts
        flagged_topics = learner_data.get('flagged_topics', [])
        redo_topics = learner_data.get('redo_topics', [])
        
        # Convert to counts for the prediction models (backward compatibility)
        redo_topics_count = len(redo_topics) if isinstance(redo_topics, list) else redo_topics
        flagged_topics_count = len(flagged_topics) if isinstance(flagged_topics, list) else flagged_topics
        
        # Identify critical topics (appearing in both lists)
        critical_topics = []
        if isinstance(flagged_topics, list) and isinstance(redo_topics, list):
            critical_topics = list(set(flagged_topics) & set(redo_topics))
        
        # 2. Predict topic revisit first (using avg_objective_score and redo_topics count)
        revisit_result = predict_topic_revisit_with_confidence(
            learner_data['avg_objective_score'], 
            redo_topics_count
        )
        
        # 2. Determine topic context based on revisit result
        if revisit_result['revisit_needed']:
            # When revisiting, use the specific current topic name
            topic_context = f"current topic: {topic_name}"
            topic_instruction = "current_topic"
            topic_for_prompt = f"current topic: {topic_name}"
        else:
            # When progressing, use generic "next topic"
            topic_context = "next topic"
            topic_instruction = "next_topic"
            topic_for_prompt = "the next topic"
        
        # 3. Predict learner category and get category prompt
        category_result = predict_learner_category(learner_data, topic_for_prompt)
        
        # 4. Use revisit result to determine revisiting_module for strategy prediction
        strategy_input = strategy_data.copy()
        strategy_input['revisiting_module'] = revisit_result['revisit_needed']
        
        # 5. Predict strategy
        strategy_result = predict_strategy(
            strategy_input['revisiting_module'],
            strategy_input['session_preference'],
            strategy_input['attention_span'],
            strategy_input['format_preference']
        )
        
        # 6. Generate user category description
        user_description = get_user_category_description(
            strategy_input['revisiting_module'],
            strategy_input['session_preference'],
            strategy_input['attention_span'],
            strategy_input['format_preference']
        )
        
        # 7. Combine category and strategy prompts with topic focus
        combined_prompt = self._combine_prompts(
            category_result['prompt'],
            strategy_result['strategy_prompt'],
            revisit_result['revisit_needed'],
            topic_instruction,
            topic_name if revisit_result['revisit_needed'] else None,
            flagged_topics,
            redo_topics,
            critical_topics
        )
        
        # 8. Create structured JSON output
        result = {
            "analysis_summary": {
                "topic_context": topic_context,
                "topic_instruction": topic_instruction,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            },
            "learner_profile": {
                "category": category_result['category'],
                "category_prompt": category_result['prompt'],
                "confidence_score": round(float(category_result['confidence']), 3),
                "redo_topics_flag": bool(category_result.get('redo_topics_flag', False)),
                "input_data": {
                    "avg_objective_score": float(learner_data['avg_objective_score']),
                    "avg_confidence_score": float(learner_data['avg_confidence_score']),
                    "learner_level": str(learner_data['learner_level']),
                    "learner_purpose": str(learner_data['learner_purpose']),
                    "redo_topics": redo_topics if isinstance(redo_topics, list) else int(redo_topics),
                    "flagged_topics": flagged_topics if isinstance(flagged_topics, list) else int(flagged_topics),
                    "critical_topics": critical_topics
                }
            },
            "topic_revisit_analysis": {
                "revisit_needed": bool(revisit_result['revisit_needed']),
                "confidence_score": round(float(revisit_result['confidence']), 3),
                "reasoning": "Based on avg_objective_score and redo_topics count"
            },
            "strategy_recommendation": {
                "strategy_label": strategy_result['strategy_label'],
                "strategy_prompt": strategy_result['strategy_prompt'],
                "confidence_score": round(float(strategy_result['confidence']), 3),
                "user_description": user_description,
                "input_parameters": {
                    "revisiting_module": bool(strategy_input['revisiting_module']),
                    "session_preference": str(strategy_input['session_preference']),
                    "attention_span": float(strategy_input['attention_span']),
                    "format_preference": str(strategy_input['format_preference'])
                }
            },
            "combined_output": {
                "final_prompt": combined_prompt,
                "teaching_approach": self._get_teaching_approach(
                    category_result['category'],
                    strategy_result['strategy_label'],
                    revisit_result['revisit_needed'],
                    flagged_topics,
                    redo_topics,
                    critical_topics
                )
            }
        }
        
        # Convert all numpy types to JSON serializable types
        result = self._convert_to_json_serializable(result)
        
        return result
    
    def _combine_prompts(self, category_prompt, strategy_prompt, revisit_needed, topic_instruction, topic_name, flagged_topics=None, redo_topics=None, critical_topics=None):
        """
        Combine category and strategy prompts into a unified prompt with topic focus.
        """
        # Ensure topic lists are valid
        flagged_topics = flagged_topics or []
        redo_topics = redo_topics or []
        critical_topics = critical_topics or []
        
        # Determine topic reference
        if revisit_needed:
            topic_ref = f"CURRENT TOPIC (Revisiting): {topic_name}"
            context_note = f"This learner is revisiting '{topic_name}' - material they've seen before."
        else:
            topic_ref = "NEXT TOPIC (New Material): Ready to progress"
            context_note = "This learner is ready to progress to the next topic in their learning path."
        
        combined = f"{topic_ref}\n\n"
        combined += f"CONTEXT: {context_note}\n\n"
        combined += f"LEARNER PROFILE: {category_prompt}\n\n"
        combined += f"DELIVERY STRATEGY: {strategy_prompt}\n\n"
        
        # Add topic-specific focus areas
        if critical_topics:
            combined += f"🚨 CRITICAL ATTENTION REQUIRED: The following topics appear in BOTH flagged and redo lists, requiring immediate focused intervention:\n"
            for topic in critical_topics:
                combined += f"   • {topic} (PRIORITY: Extra reinforcement needed)\n"
            combined += "\n"
        
        if flagged_topics:
            combined += f"⚠️ FLAGGED TOPICS requiring special attention:\n"
            for topic in flagged_topics:
                if topic not in critical_topics:  # Don't repeat critical topics
                    combined += f"   • {topic}\n"
            combined += "\n"
        
        if redo_topics:
            combined += f"🔄 REDO TOPICS needing revisitation:\n"
            for topic in redo_topics:
                if topic not in critical_topics:  # Don't repeat critical topics
                    combined += f"   • {topic}\n"
            combined += "\n"
        
        if revisit_needed:
            combined += f"IMPORTANT: Focus on reinforcing and clarifying concepts from '{topic_name}'. "
            combined += "Address gaps in understanding and build stronger foundations before moving forward."
            if flagged_topics or redo_topics:
                combined += " Pay special attention to the flagged and redo topics listed above."
            combined += "\n\n"
        else:
            combined += "IMPORTANT: Introduce new concepts for the next topic in their learning sequence. "
            combined += "Ensure prerequisites are met before advancing to new material."
            if flagged_topics or redo_topics:
                combined += " Before progressing, ensure understanding of the flagged and redo topics listed above."
            combined += "\n\n"
        
        combined += "FINAL INSTRUCTION: Create personalized learning content by combining the learner "
        combined += "profile guidance with the delivery strategy, keeping the topic context in mind."
        
        return combined
    
    def _get_teaching_approach(self, category, strategy, revisit_needed, flagged_topics=None, redo_topics=None, critical_topics=None):
        """
        Generate a structured teaching approach summary with topic-specific guidance.
        """
        # Ensure topic lists are valid
        flagged_topics = flagged_topics or []
        redo_topics = redo_topics or []
        critical_topics = critical_topics or []
        
        approach = {
            "learner_type": str(category),
            "delivery_method": str(strategy),
            "content_focus": "Reinforcement and clarification" if revisit_needed else "New concept introduction",
            "pacing": self._get_pacing_recommendation(category),
            "support_level": self._get_support_level(category),
            "challenge_level": self._get_challenge_level(category)
        }
        
        # Add topic-specific guidance
        topic_guidance = []
        
        if critical_topics:
            topic_guidance.append({
                "priority": "CRITICAL",
                "topics": critical_topics,
                "action": "Immediate focused intervention required - these topics appear in both flagged and redo lists",
                "approach": "Extra reinforcement, multiple explanations, hands-on practice, frequent assessment"
            })
        
        if flagged_topics:
            non_critical_flagged = [t for t in flagged_topics if t not in critical_topics]
            if non_critical_flagged:
                topic_guidance.append({
                    "priority": "HIGH",
                    "topics": non_critical_flagged,
                    "action": "Special attention required - learner struggled with these concepts",
                    "approach": "Additional examples, alternative explanations, scaffolded practice"
                })
        
        if redo_topics:
            non_critical_redo = [t for t in redo_topics if t not in critical_topics]
            if non_critical_redo:
                topic_guidance.append({
                    "priority": "MEDIUM",
                    "topics": non_critical_redo,
                    "action": "Revisitation needed - concepts require reinforcement",
                    "approach": "Review, practice, application in new contexts"
                })
        
        if topic_guidance:
            approach["topic_specific_guidance"] = topic_guidance
        
        return approach
    
    def _get_pacing_recommendation(self, category):
        """Get pacing recommendation based on learner category."""
        pacing_map = {
            'Struggling Novice': 'Very slow with frequent breaks',
            'Hesitant Learner': 'Slow with gentle progression',
            'Overconfident Novice': 'Moderate with reality checks',
            'Anxious Improver': 'Steady with confidence building',
            'Rising Improver': 'Moderate with momentum maintenance',
            'Overreacher': 'Controlled with structured guidance',
            'Confidence-Delayed': 'Standard with independence building',
            'Steady Performer': 'Standard with consistent progression',
            'Confident Achiever': 'Flexible with varied challenges',
            'Humble Expert': 'Advanced with nuanced discussions',
            'Stable Expert': 'Fast with minimal guidance',
            'Imposter Syndrome': 'Standard with competence affirmation'
        }
        return pacing_map.get(category, 'Standard')
    
    def _get_support_level(self, category):
        """Get support level based on learner category."""
        support_map = {
            'Struggling Novice': 'High support with frequent encouragement',
            'Hesitant Learner': 'High support with reassurance',
            'Overconfident Novice': 'Moderate support with gentle correction',
            'Anxious Improver': 'High support with positive reinforcement',
            'Rising Improver': 'Moderate support with progress tracking',
            'Overreacher': 'Moderate support with reality grounding',
            'Confidence-Delayed': 'Low support with independence focus',
            'Steady Performer': 'Standard support with consistency',
            'Confident Achiever': 'Low support with exploration freedom',
            'Humble Expert': 'Minimal support with growth encouragement',
            'Stable Expert': 'Minimal support with innovation focus',
            'Imposter Syndrome': 'Moderate support with competence validation'
        }
        return support_map.get(category, 'Standard support')
    
    def _get_challenge_level(self, category):
        """Get challenge level based on learner category."""
        challenge_map = {
            'Struggling Novice': 'Very low with achievable milestones',
            'Hesitant Learner': 'Low with gentle scaffolding',
            'Overconfident Novice': 'Moderate with misconception addressing',
            'Anxious Improver': 'Low-moderate with confidence building',
            'Rising Improver': 'Moderate with steady advancement',
            'Overreacher': 'Moderate with structured limits',
            'Confidence-Delayed': 'Moderate-high with competence emphasis',
            'Steady Performer': 'Moderate with consistent challenges',
            'Confident Achiever': 'High with varied applications',
            'Humble Expert': 'High with advanced concepts',
            'Stable Expert': 'Very high with cutting-edge content',
            'Imposter Syndrome': 'High with expertise recognition'
        }
        return challenge_map.get(category, 'Moderate')
    
    def get_json_summary(self, result):
        """
        Generate a JSON summary of the combined prediction results.
        """
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def print_formatted_result(self, result):
        """
        Print the result in a nicely formatted way.
        """
        print("=" * 80)
        print("COMBINED LEARNER ANALYSIS - JSON FORMAT")
        print("=" * 80)
        try:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            print("Raw result:", result)
        print("=" * 80)

# === Example usage ===
if __name__ == "__main__":
    # Initialize the combined system
    system = CombinedLearnerSystem()
    
    # Define example learner data
    learner_data = {
        'avg_objective_score': 6.0,
        'avg_confidence_score': 0.45,
        'confidence_trend': 0.02,
        'avg_skill_score': 4.5,
        'learner_level': 'intermediate',
        'learner_purpose': 'scratch',
        'flagged_topics': ['variables', 'loops', 'functions'],
        'redo_topics': ['loops', 'conditionals', 'data structures'],
        'stddev_objective_score': 0.6
    }
    
    # Define strategy parameters
    strategy_data = {
        'session_preference': 'short_chunks',
        'attention_span': 8.5,
        'format_preference': 'text'
    }
    
    # Make combined prediction
    result = system.predict_combined(
        learner_data, 
        strategy_data, 
        "Python object-oriented programming"
    )
    
    # Display results in JSON format
    print("=== EXAMPLE 1: Learner with High Redo Topics ===")
    system.print_formatted_result(result)
    
    # Example 2: Different learner type
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced Learner with Low Redo Topics")
    print("="*80)
    
    learner_data2 = {
        'avg_objective_score': 8.5,
        'avg_confidence_score': 0.75,
        'confidence_trend': 0.05,
        'avg_skill_score': 6.8,
        'learner_level': 'advanced',
        'learner_purpose': 'exploratory',
        'flagged_topics': ['neural networks', 'optimization'],
        'redo_topics': [],
        'stddev_objective_score': 0.1
    }
    
    strategy_data2 = {
        'session_preference': 'long_sessions',
        'attention_span': 20.0,
        'format_preference': 'video'
    }
    
    result2 = system.predict_combined(
        learner_data2,
        strategy_data2,
        "Advanced machine learning algorithms"
    )
    
    system.print_formatted_result(result2)
    
    # Example 3: Save results to JSON file
    print("\n" + "="*80)
    print("SAVING RESULTS TO JSON FILES")
    print("="*80)
    
    # Save individual results
    with open("learner_analysis_example1.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    with open("learner_analysis_example2.json", "w") as f:
        json.dump(result2, f, indent=2, ensure_ascii=False)
    
    print("✅ Results saved to:")
    print("- learner_analysis_example1.json")
    print("- learner_analysis_example2.json")