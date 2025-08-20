# Topic Priority Management System - Complete Implementation

## Overview

This system enhances the decision tree outputs by adding intelligent topic priority tracking across modules with weakness identification prompts. It implements a sliding window approach to track user difficulties and provide targeted remediation suggestions.

## System Architecture

### Core Components

1. **TopicPriorityManager** - Main class for topic tracking and categorization
2. **Enhanced Decision Tree Integration** - Updated test system with skill analysis
3. **Weakness Prompt Generation** - Automated user weakness identification
4. **Sliding Window Tracking** - 3-module rolling history management

## Topic Categorization Logic

### Critical Topics (3-module tracking)
- **Condition**: Topic appears in BOTH redo_topics AND flagged_topics 
- **Duration**: Tracked for 3 consecutive modules
- **Purpose**: Identifies persistent, severe learning difficulties
- **Example**: Student redoes "loops" AND flags "loops" → Critical weakness

### High Priority Topics (1-module tracking)  
- **Condition**: Topic appears in EITHER redo_topics OR flagged_topics (but not both)
- **Duration**: Tracked for 1 module only
- **Purpose**: Identifies temporary or specific learning challenges
- **Example**: Student only redoes "arrays" → High priority for next module

## Implementation Details

### Data Flow
```
Raw User Data → Topic Categorization → Priority Tracking → Weakness Prompts → Decision Tree Integration
```

### Key Features
- **Automatic Expiration**: Topics expire after their tracking duration
- **Priority Override**: Critical topics take precedence over high priority
- **Window Management**: Only last 3 modules stored in memory
- **Dynamic Updates**: Real-time topic priority adjustments

## Input Format

Based on the sample from `pre.py`:
```python
sample_input = {
    "module_id": "AAC",
    "user_id": "user_12345", 
    "redo_topics": ["topic a", "topic b"],      # Topics student redid
    "flagged_topics": ["topic b", "topic c"],   # Topics student flagged as difficult
    # ... other decision tree features
}
```

## Output Structure

### Enhanced Decision Tree Output
```json
{
  "user_id": "U001",
  "module_id": "M1", 
  "user_category": {
    "mismatch_pattern": "Perfect Alignment",
    "personalization_strategy": "Advanced content, accelerated pace",
    "skill_analysis": {
      "strongest_skill": "coding",
      "skill_statement": "User is master in coding (0.90) - strongest area"
    }
  },
  "redo_decision": {"decision": "NO", "confidence": 1.0},
  "learning_intervention": {"intervention": "Extended Examples"},
  "topic_priorities": {
    "current_priority_topics": {
      "critical": ["topic_b"],
      "high": ["topic_a", "topic_c"]
    },
    "weakness_prompts": {
      "critical": "User shows critical weakness in: topic_b. These topics require immediate attention and will be reinforced across multiple modules.",
      "high": "User shows high priority weakness in: topic_a, topic_c. These topics need focused review in the current module."
    },
    "topic_details": {
      "critical_topics": ["topic_b"],
      "high_priority_topics": ["topic_a", "topic_c"], 
      "total_tracked": 3
    }
  }
}
```

## Weakness Prompt Examples

### Critical Weakness Prompts
- "User shows critical weakness in: loops, functions. These topics require immediate attention and will be reinforced across multiple modules."
- "User shows critical weakness in: data structures. This topic requires immediate attention and will be reinforced across multiple modules."

### High Priority Prompts  
- "User shows high priority weakness in: arrays, sorting. These topics need focused review in the current module."
- "User shows high priority weakness in: recursion. This topic needs focused review in the current module."

## Usage Examples

### Basic Usage
```python
from topic_priority_manager import TopicPriorityManager

# Initialize
topic_manager = TopicPriorityManager(window_size=3)

# Process module
topic_manager.update_topic_tracker(
    module_id="M1",
    redo_topics=["loops", "variables"], 
    flagged_topics=["loops", "functions"]
)

# Get analysis
summary = topic_manager.get_topic_summary()
weakness_prompts = summary['weakness_prompts']
```

### Integration with Decision Trees
```python
# Enhanced test with topic priorities
results = []
topic_manager = TopicPriorityManager(window_size=3)

for user in sample_users:
    # Update topic tracking
    topic_manager.update_topic_tracker(
        user["module_id"], 
        user["redo_topics"],
        user["flagged_topics"] 
    )
    
    # Get decision tree predictions + topic analysis
    user_result = get_enhanced_predictions(user, topic_manager)
    results.append(user_result)
```

## Files in Implementation

### Core Files
- **`topic_priority_manager.py`** - Main topic tracking implementation
- **`test_decision_tree_outputs.py`** - Enhanced test system with topic integration  
- **`demonstrate_topic_system.py`** - Comprehensive demonstration script

### Model Files (Updated)
- **`use_user_categorization_model.py`** - Enhanced with skill analysis
- **`use_redo_decision_model.py`** - Redo decision predictions
- **`use_learning_intervention_model.py`** - Learning intervention recommendations

### Output Files
- **`test_decision_tree_outputs.json`** - Complete system output with topic priorities
- **`topic_priority_demonstration.json`** - Detailed demonstration results  
- **`topic_priority_results.json`** - Standalone topic tracking results

## Key Algorithms

### Topic Categorization Algorithm
```python
def categorize_topics(redo_topics, flagged_topics):
    redo_set = set(redo_topics)
    flagged_set = set(flagged_topics) 
    
    # Critical: intersection (in both lists)
    critical = list(redo_set.intersection(flagged_set))
    
    # High: symmetric difference (in either but not both)  
    high = list(redo_set.symmetric_difference(flagged_set))
    
    return {'critical': critical, 'high': high}
```

### Sliding Window Management
```python
def update_topic_tracker(self, module_id, redo_topics, flagged_topics):
    # Add new topics with appropriate durations
    for topic in critical_topics:
        self.topic_tracker[topic] = {
            'priority': 'critical',
            'remaining_modules': 3,  # Track for 3 modules
            'first_seen_module': module_id
        }
    
    # Decrement all tracked topics
    for topic in self.topic_tracker:
        self.topic_tracker[topic]['remaining_modules'] -= 1
    
    # Remove expired topics
    self.topic_tracker = {
        topic: data for topic, data in self.topic_tracker.items() 
        if data['remaining_modules'] > 0
    }
```

## Benefits

1. **Personalized Learning**: Identifies specific user weaknesses
2. **Persistence Tracking**: Distinguishes between temporary and chronic difficulties  
3. **Automated Prompts**: Generates contextual weakness messages
4. **Resource Optimization**: Focuses remediation on high-impact topics
5. **Scalable Design**: Easily handles multiple users and modules
6. **Integration Ready**: Works seamlessly with existing decision tree models

## Testing & Validation

The system has been tested with:
- ✅ 5 sample users across 5 modules
- ✅ Various topic combinations (critical, high priority, mixed)
- ✅ Sliding window expiration scenarios  
- ✅ Integration with all 3 decision tree models
- ✅ Skill analysis integration
- ✅ Weakness prompt generation

## Future Enhancements

1. **Adaptive Window Size**: Dynamic tracking duration based on user performance
2. **Topic Similarity**: Group related topics for better tracking
3. **Intervention Prioritization**: Rank topics by urgency and impact
4. **Progress Tracking**: Monitor improvement over time  
5. **Batch Processing**: Handle multiple users simultaneously

---

*This system successfully addresses the requirement to track user weaknesses across modules with intelligent categorization and automated prompt generation, fully integrated with the existing decision tree infrastructure.*
