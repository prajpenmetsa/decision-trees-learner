#!/usr/bin/env python3
"""
Demonstration script showing the topic priority management system
with enhanced decision tree outputs including skill analysis.

This script demonstrates:
1. Topic categorization (Critical vs High priority)
2. Sliding window tracking across modules
3. Weakness prompt generation
4. Integration with decision tree models
"""

import json
import os
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from topic_priority_manager import TopicPriorityManager

def demonstrate_topic_priority_system():
    """
    Comprehensive demonstration of the topic priority system with various scenarios.
    """
    print("=" * 80)
    print("TOPIC PRIORITY MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis system tracks user weaknesses across modules using a sliding window approach:")
    print("• CRITICAL topics (in both redo & flagged): Tracked for 3 modules")
    print("• HIGH PRIORITY topics (in either redo or flagged): Tracked for 1 module")
    print("• Weakness prompts are generated based on currently active topics")
    print("\n" + "=" * 80)
    
    # Initialize topic manager
    topic_manager = TopicPriorityManager(window_size=3)
    
    # Sample data with diverse topic scenarios
    sample_scenarios = [
        {
            "module_id": "M1_Intro_Programming", 
            "user_id": "student_001",
            "redo_topics": ["variables", "loops"],
            "flagged_topics": ["loops", "functions"],
            "description": "Student struggles with loops (appears in both lists - CRITICAL)"
        },
        {
            "module_id": "M2_Data_Structures", 
            "user_id": "student_001",
            "redo_topics": ["arrays"],
            "flagged_topics": ["dictionaries"],
            "description": "Different topics in redo vs flagged - both HIGH priority"
        },
        {
            "module_id": "M3_Algorithms", 
            "user_id": "student_001",
            "redo_topics": ["sorting", "loops"],  # loops again - will be critical
            "flagged_topics": ["loops", "searching"],
            "description": "Loops reappears (CRITICAL), sorting only flagged (HIGH)"
        },
        {
            "module_id": "M4_Advanced_Topics", 
            "user_id": "student_001",
            "redo_topics": [],
            "flagged_topics": ["recursion"],
            "description": "Only new topic flagged (HIGH priority)"
        },
        {
            "module_id": "M5_Final_Review", 
            "user_id": "student_001",
            "redo_topics": ["objects"],
            "flagged_topics": [],
            "description": "Only new topic in redo (HIGH priority)"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(sample_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"MODULE {i}: {scenario['module_id']}")
        print(f"{'='*60}")
        print(f"Description: {scenario['description']}")
        print(f"Redo topics: {scenario['redo_topics']}")
        print(f"Flagged topics: {scenario['flagged_topics']}")
        
        # Update topic manager
        topic_manager.update_topic_tracker(
            scenario['module_id'],
            scenario['redo_topics'], 
            scenario['flagged_topics']
        )
        
        # Get current analysis
        topic_summary = topic_manager.get_topic_summary()
        
        print(f"\nCURRENT ANALYSIS:")
        print(f"Critical topics: {topic_summary['current_priority_topics']['critical']}")
        print(f"High priority topics: {topic_summary['current_priority_topics']['high']}")
        print(f"Total tracked topics: {topic_summary['total_tracked_topics']}")
        
        # Display weakness prompts
        if topic_summary['weakness_prompts']:
            print(f"\nWEAKNESS PROMPTS:")
            for category, prompt in topic_summary['weakness_prompts'].items():
                print(f"  {category.upper()}: {prompt}")
        else:
            print(f"\nNo weakness prompts (no active priority topics)")
        
        # Show topic tracker details
        if topic_summary['topic_tracker_details']:
            print(f"\nTOPIC TRACKER DETAILS:")
            for topic, details in topic_summary['topic_tracker_details'].items():
                remaining = details['remaining_modules']
                priority = details['priority'].upper()
                first_seen = details['first_seen_module']
                print(f"  • {topic}: {priority} priority, {remaining} modules remaining (first seen: {first_seen})")
        
        # Store result
        result = {
            'module': scenario['module_id'],
            'user_id': scenario['user_id'],
            'input': {
                'redo_topics': scenario['redo_topics'],
                'flagged_topics': scenario['flagged_topics']
            },
            'analysis': topic_summary
        }
        results.append(result)
        
        print(f"\n{'-'*40}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SYSTEM STATE")
    print(f"{'='*80}")
    
    final_summary = topic_manager.get_topic_summary()
    print(f"Total modules processed: {len(sample_scenarios)}")
    print(f"Currently tracked topics: {final_summary['total_tracked_topics']}")
    
    if final_summary['current_priority_topics']['critical']:
        print(f"Active critical topics: {final_summary['current_priority_topics']['critical']}")
    if final_summary['current_priority_topics']['high']:
        print(f"Active high priority topics: {final_summary['current_priority_topics']['high']}")
    
    if final_summary['weakness_prompts']:
        print(f"\nFINAL WEAKNESS PROMPTS:")
        for category, prompt in final_summary['weakness_prompts'].items():
            print(f"  {category.upper()}: {prompt}")
    
    # Show module history (sliding window)
    print(f"\nMODULE HISTORY (last {topic_manager.window_size} modules):")
    for i, module in enumerate(final_summary['module_history'], 1):
        print(f"  {i}. {module['module_id']}")
        print(f"     Critical: {module['critical']}")
        print(f"     High: {module['high']}")
    
    # Save results
    output_file = "topic_priority_demonstration.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results, topic_manager

def show_integration_example():
    """
    Show how the topic priority system integrates with decision tree outputs.
    """
    print(f"\n{'='*80}")
    print("INTEGRATION WITH DECISION TREE MODELS")
    print(f"{'='*80}")
    print("\nThe topic priority system is now integrated with decision tree outputs.")
    print("Each user's result includes:")
    print("  • User categorization with skill analysis")
    print("  • Redo decision recommendation") 
    print("  • Learning intervention suggestion")
    print("  • Topic priority tracking with weakness prompts")
    print("\nExample structure in test_decision_tree_outputs.json:")
    
    example_structure = {
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
                "critical": ["topic_a", "topic_b"],
                "high": ["topic_c"]
            },
            "weakness_prompts": {
                "critical": "User shows critical weakness in: topic_a, topic_b. These topics require immediate attention and will be reinforced across multiple modules."
            }
        }
    }
    
    print(json.dumps(example_structure, indent=2))

if __name__ == "__main__":
    # Run the demonstration
    results, topic_manager = demonstrate_topic_priority_system()
    
    # Show integration example
    show_integration_example()
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*80}")
    print("\nKey Features Demonstrated:")
    print("✓ Critical topic identification (topics in both redo & flagged)")
    print("✓ High priority topic tracking (topics in either list)")
    print("✓ Sliding window management (3-module tracking)")
    print("✓ Automatic topic expiration")
    print("✓ Weakness prompt generation")
    print("✓ Integration with decision tree models")
    print("✓ Comprehensive user analysis")
    
    print(f"\nFiles Generated:")
    print(f"• topic_priority_demonstration.json - Detailed demonstration results")
    print(f"• test_decision_tree_outputs.json - Complete system output with all features")
