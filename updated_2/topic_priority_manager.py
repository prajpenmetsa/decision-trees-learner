import json
from typing import Dict, List, Any, Set
from collections import defaultdict, deque

class TopicPriorityManager:
    """
    Manages topic priorities across modules with sliding window tracking.
    
    Critical topics (in both redo and flagged): Show for 3 modules
    High priority topics (in either redo or flagged): Show for 1 module
    """
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        # Track topics with their priority and remaining modules to show
        self.topic_tracker = {}
        # Module history for sliding window (stores last 3 modules)
        self.module_history = deque(maxlen=window_size)
    
    def categorize_topics(self, redo_topics: List[str], flagged_topics: List[str]) -> Dict[str, List[str]]:
        """
        Categorize topics based on their presence in redo and flagged lists.
        
        Args:
            redo_topics: List of topics that were redone
            flagged_topics: List of topics that were flagged
            
        Returns:
            Dictionary with 'critical' and 'high' topic lists
        """
        redo_set = set(redo_topics) if redo_topics else set()
        flagged_set = set(flagged_topics) if flagged_topics else set()
        
        # Critical: Topics in both redo and flagged
        critical_topics = list(redo_set.intersection(flagged_set))
        
        # High: Topics in either redo or flagged (but not both)
        high_topics = list(redo_set.symmetric_difference(flagged_set))
        
        return {
            'critical': critical_topics,
            'high': high_topics
        }
    
    def update_topic_tracker(self, current_module_id: str, redo_topics: List[str], flagged_topics: List[str]):
        """
        Update the topic tracker with current module data.
        
        Args:
            current_module_id: ID of current module
            redo_topics: Topics redone in current module
            flagged_topics: Topics flagged in current module
        """
        # First, decrement remaining modules for existing tracked topics
        topics_to_remove = []
        for topic, data in self.topic_tracker.items():
            data['remaining_modules'] -= 1
            if data['remaining_modules'] <= 0:
                topics_to_remove.append(topic)
        
        # Remove expired topics
        for topic in topics_to_remove:
            del self.topic_tracker[topic]
        
        # Get categorized topics for current module
        categorized = self.categorize_topics(redo_topics, flagged_topics)
        
        # Add current module to history
        module_data = {
            'module_id': current_module_id,
            'critical': categorized['critical'],
            'high': categorized['high']
        }
        self.module_history.append(module_data)
        
        # Add critical topics (show for 3 modules)
        for topic in categorized['critical']:
            self.topic_tracker[topic] = {
                'priority': 'critical',
                'remaining_modules': 3,  # Will show for 3 full modules
                'first_seen_module': current_module_id
            }
        
        # Add high priority topics (show for next 1 module)
        for topic in categorized['high']:
            if topic not in self.topic_tracker:  # Don't override critical topics
                self.topic_tracker[topic] = {
                    'priority': 'high',
                    'remaining_modules': 1,  # Will show for 1 full module
                    'first_seen_module': current_module_id
                }
            elif self.topic_tracker[topic]['priority'] == 'high':
                # Reset remaining modules if topic appears again as high priority
                self.topic_tracker[topic]['remaining_modules'] = 1
        
        print(f"Debug - Module {current_module_id}:")
        print(f"  Redo topics: {redo_topics}")
        print(f"  Flagged topics: {flagged_topics}")
        print(f"  Critical topics (both lists): {categorized['critical']}")
        print(f"  High priority topics (symmetric diff): {categorized['high']}")
        print(f"  Currently tracked topics: {list(self.topic_tracker.keys())}")
        
        # Show detailed tracking info
        for topic, data in self.topic_tracker.items():
            print(f"    {topic}: {data['priority']} priority, {data['remaining_modules']} modules remaining")
        print("-" * 50)
    
    def get_current_priority_topics(self) -> Dict[str, List[str]]:
        """
        Get currently active priority topics.
        
        Returns:
            Dictionary with 'critical' and 'high' topic lists currently being tracked
        """
        current_critical = []
        current_high = []
        
        for topic, data in self.topic_tracker.items():
            if data['remaining_modules'] > 0:
                if data['priority'] == 'critical':
                    current_critical.append(topic)
                elif data['priority'] == 'high':
                    current_high.append(topic)
        
        return {
            'critical': current_critical,
            'high': current_high
        }
    
    def generate_weakness_prompt(self) -> Dict[str, str]:
        """
        Generate prompts about user weaknesses based on tracked topics.
        
        Returns:
            Dictionary with prompts for critical and high priority topics
        """
        current_topics = self.get_current_priority_topics()
        
        prompts = {}
        
        if current_topics['critical']:
            critical_list = ", ".join(current_topics['critical'])
            prompts['critical'] = f"User shows critical weakness in: {critical_list}. These topics require immediate attention and will be reinforced across multiple modules."
        
        if current_topics['high']:
            high_list = ", ".join(current_topics['high'])
            prompts['high'] = f"User shows high priority weakness in: {high_list}. These topics need focused review in the current module."
        
        return prompts
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of topic tracking status.
        
        Returns:
            Dictionary with current status, history, and prompts
        """
        current_topics = self.get_current_priority_topics()
        weakness_prompts = self.generate_weakness_prompt()
        
        return {
            'current_priority_topics': current_topics,
            'weakness_prompts': weakness_prompts,
            'topic_tracker_details': dict(self.topic_tracker),
            'module_history': list(self.module_history),
            'total_tracked_topics': len(self.topic_tracker)
        }

def process_user_with_topic_priorities(user_data: Dict[str, Any], topic_manager: TopicPriorityManager) -> Dict[str, Any]:
    """
    Process a single user's data and update topic priorities.
    
    Args:
        user_data: User data dictionary
        topic_manager: TopicPriorityManager instance
        
    Returns:
        Dictionary with user results and topic priority information
    """
    # Update topic manager with current user data
    topic_manager.update_topic_tracker(
        user_data.get('module_id', ''),
        user_data.get('redo_topics', []),
        user_data.get('flagged_topics', [])
    )
    
    # Get topic summary
    topic_summary = topic_manager.get_topic_summary()
    
    return {
        'user_id': user_data.get('user_id', ''),
        'module_id': user_data.get('module_id', ''),
        'topic_priorities': topic_summary
    }

# Example usage and testing
def main():
    print("Topic Priority Manager - Testing with Sample Data")
    print("=" * 60)
    
    # Initialize topic manager
    topic_manager = TopicPriorityManager(window_size=3)
    
    # Sample data simulating 4 modules for a user
    sample_modules = [
        {
            "module_id": "M1", "user_id": "user_001",
            "redo_topics": ["topic_a", "topic_b"],
            "flagged_topics": ["topic_b", "topic_c"]
        },
        {
            "module_id": "M2", "user_id": "user_001", 
            "redo_topics": ["topic_d"],
            "flagged_topics": ["topic_a", "topic_d"]
        },
        {
            "module_id": "M3", "user_id": "user_001",
            "redo_topics": ["topic_e"],
            "flagged_topics": ["topic_f"]
        },
        {
            "module_id": "M4", "user_id": "user_001",
            "redo_topics": [],
            "flagged_topics": ["topic_g"]
        }
    ]
    
    results = []
    
    for i, module_data in enumerate(sample_modules, 1):
        print(f"\nProcessing Module {i}: {module_data['module_id']}")
        print(f"Redo topics: {module_data['redo_topics']}")
        print(f"Flagged topics: {module_data['flagged_topics']}")
        
        # Process module
        result = process_user_with_topic_priorities(module_data, topic_manager)
        results.append(result)
        
        # Show current status
        topic_summary = result['topic_priorities']
        print(f"\nCurrent Priority Topics:")
        print(f"  Critical: {topic_summary['current_priority_topics']['critical']}")
        print(f"  High: {topic_summary['current_priority_topics']['high']}")
        
        if topic_summary['weakness_prompts']:
            print(f"\nWeakness Prompts:")
            for category, prompt in topic_summary['weakness_prompts'].items():
                print(f"  {category.title()}: {prompt}")
        else:
            print(f"\nNo current weakness prompts.")
        
        print(f"Total tracked topics: {topic_summary['total_tracked_topics']}")
        print("-" * 40)
    
    # Save results
    with open('topic_priority_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'topic_priority_results.json'")
    
    # Show final tracker state
    print(f"\nFinal Topic Tracker State:")
    final_summary = topic_manager.get_topic_summary()
    print(json.dumps(final_summary, indent=2))

if __name__ == "__main__":
    main()
