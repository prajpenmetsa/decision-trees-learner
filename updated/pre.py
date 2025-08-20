import json
import statistics
from typing import Dict, List, Any

class DecisionTreePreprocessor:
    """
    Simple preprocessor to calculate required features from raw input data.
    """
    
    def calculate_avg_objective_score(self, objective_scores: List[float]) -> float:
        """Calculate average objective score from topic scores."""
        if not objective_scores:
            return 0.0
        return sum(objective_scores) / len(objective_scores)
    
    def calculate_avg_confidence_score(self, confidence_scores: List[float]) -> float:
        """Calculate average confidence score from topic scores."""
        if not confidence_scores:
            return 0.0
        return sum(confidence_scores) / len(confidence_scores)
    
    def calculate_avg_time_module(self, time_module: List[float]) -> float:
        """Calculate average time spent on module."""
        if not time_module:
            return 0.0
        return sum(time_module) / len(time_module)
    
    def calculate_confidence_trend(self, confidence_previous: List[float]) -> float:
        """
        Calculate confidence trend from previous module confidence scores.
        Returns a value between -1 and 1:
        - Positive values (0 to 1): Confidence is increasing over time
        - Negative values (-1 to 0): Confidence is decreasing over time
        - 0: No clear trend or insufficient data
        """
        if len(confidence_previous) < 2:
            return 0.0
        
        # Calculate linear trend using simple slope calculation
        n = len(confidence_previous)
        x_vals = list(range(n))
        
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(confidence_previous)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, confidence_previous))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize to [-1, 1] range
        # Adjust max_slope based on realistic confidence score changes per module
        # Assuming confidence scores are between 0-1 and we have at most 10 modules
        max_slope = 0.1  # This means a change of 0.1 per module would give trend = 1 or -1
        normalized_trend = max(-1.0, min(1.0, slope / max_slope))
        
        return normalized_trend
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw input into required features.
        
        Args:
            input_data: Raw input dictionary
            
        Returns:
            Dictionary with calculated features
        """
        
        # Print input features before processing
        print("=" * 60)
        print("INPUT FEATURES (Before Processing):")
        print("=" * 60)
        for key, value in input_data.items():
            if isinstance(value, list) and len(value) > 5:
                # For long lists, show first few items and count
                print(f"{key}: [{', '.join(map(str, value[:3]))}, ...] (total: {len(value)} items)")
            else:
                print(f"{key}: {value}")
        print()
        
        # Calculate required features
        avg_obj_score = self.calculate_avg_objective_score(
            input_data.get("objective_scores", [])
        )
        
        avg_conf_score = self.calculate_avg_confidence_score(
            input_data.get("confidence_scores", [])
        )
        
        conf_trend = self.calculate_confidence_trend(
            input_data.get("confidence_previous", [])
        )

        avg_time_module = self.calculate_avg_time_module(
            input_data.get("time_module", [])
        )

        redo_topics = input_data.get("redo_topics", [])
        flagged_topics = input_data.get("flagged_topics", [])
        redo_topics_count = len(redo_topics)
        flagged_topics_count = len(flagged_topics)
        
        # Output processed features
        processed_features = {
            "module_id": input_data.get("module_id", ""),
            "user_id": input_data.get("user_id", ""),
            "avg_obj_score": round(avg_obj_score, 3),
            "avg_conf_score": round(avg_conf_score, 3),
            "conf_trend": round(conf_trend, 3),
            "redo_topics_count": redo_topics_count,
            "flagged_topics_count": flagged_topics_count,
            "redo_topics": redo_topics,
            "flagged_topics": flagged_topics,
            "skill_level": input_data.get("skill_level", {}),
            "learner_purpose": input_data.get("learner_purpose", ""),
            "desired_proficiency": input_data.get("desired_proficiency", ""),
            "session_preference": input_data.get("session_preference", ""),
            "avg_time_module": round(avg_time_module, 3),
        }
        
        print("=" * 60)
        print("PROCESSED FEATURES (After Processing):")
        print("=" * 60)
        for key, value in processed_features.items():
            print(f"{key}: {value}")
        print()
        
        return processed_features
    
    def process_and_save(self, input_data: Dict[str, Any], output_file: str = "processed_features.json") -> Dict[str, Any]:
        """Process input and save to file."""
        processed = self.preprocess_input(input_data)
        
        with open(output_file, 'w') as f:
            json.dump(processed, f, indent=2)
        
        print(f"Processed features saved to {output_file}")
        return processed

# Example usage
def main():
    sample_input = {
        "module_id": "AAC",
        "user_id": "user_12345",
        "objective_scores": [0.8, 0.7, 0.9, 0.6, 0.8],
        "confidence_scores": [0.7, 0.6, 0.5, 0.2, 0.1],
        "redo_topics": ["topic a", "topic b"],
        "flagged_topics": ["topic b", "topic c"],
        "confidence_previous": [0.6, 0.7, 0.8, 0.3, 0.2],
        "skill_level": {"logic": 0.4, "coding": 0.6, "memory": 0.5},
        "learner_purpose": "revision",
        "desired_proficiency": "intermediate",
        "session_preference": "long",
        "time_module": [300,600,800,2000,1000]
    }
    
    preprocessor = DecisionTreePreprocessor()
    
    print("DecisionTree Preprocessor - Feature Calculation")
    print("=" * 80)
    print("This preprocessor converts raw input data into features required by decision tree models.")
    print("It calculates averages, trends, and counts from raw performance data.\n")
    
    processed_features = preprocessor.preprocess_input(sample_input)
    
    print("=" * 60)
    print("CALCULATION DETAILS:")
    print("=" * 60)
    print(f"Average Objective Score: {sum(sample_input['objective_scores'])} / {len(sample_input['objective_scores'])} = {processed_features['avg_obj_score']}")
    print(f"Average Confidence Score: {sum(sample_input['confidence_scores'])} / {len(sample_input['confidence_scores'])} = {processed_features['avg_conf_score']}")
    print(f"Confidence Trend: Calculated from {len(sample_input['confidence_previous'])} previous scores = {processed_features['conf_trend']}")
    print(f"Redo Topics Count: {len(sample_input['redo_topics'])} topics = {processed_features['redo_topics_count']}")
    print(f"Flagged Topics Count: {len(sample_input['flagged_topics'])} topics = {processed_features['flagged_topics_count']}")
    print(f"Average Time per Module: {sum(sample_input['time_module'])} / {len(sample_input['time_module'])} = {processed_features['avg_time_module']} seconds")
    
    return processed_features

if __name__ == "__main__":
    main()