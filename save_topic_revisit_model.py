import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import html
import pickle

# === 1. Define data for topic revisit logic ===
revisit_data = [
    {'avg_objective_score': 5.0, 'redo_topics': 4, 'redo_flag': True},  # Struggling
    {'avg_objective_score': 4.5, 'redo_topics': 3, 'redo_flag': True},  # Struggling
    {'avg_objective_score': 6.0, 'redo_topics': 2, 'redo_flag': True},  # Developing
    {'avg_objective_score': 5.8, 'redo_topics': 3, 'redo_flag': True},  # Developing
    {'avg_objective_score': 7.0, 'redo_topics': 3, 'redo_flag': True},  # Competent
    {'avg_objective_score': 6.8, 'redo_topics': 2, 'redo_flag': False}, # Competent
    {'avg_objective_score': 7.5, 'redo_topics': 1, 'redo_flag': False}, # Competent
    {'avg_objective_score': 8.5, 'redo_topics': 0, 'redo_flag': False}, # Expert
    {'avg_objective_score': 8.2, 'redo_topics': 1, 'redo_flag': False}, # Expert
    {'avg_objective_score': 8.8, 'redo_topics': 0, 'redo_flag': False}, # Expert
]

# === 2. Create DataFrame ===
revisit_df = pd.DataFrame(revisit_data)

# === 3. Features and target ===
revisit_features = ['avg_objective_score', 'redo_topics']
revisit_target = 'redo_flag'

# === 4. Train decision tree for topic revisit logic ===
revisit_model = DecisionTreeClassifier(max_depth=3, random_state=42)
revisit_model.fit(revisit_df[revisit_features], revisit_df[revisit_target])

# === 5. Visualize using Graphviz ===
dot_data = export_graphviz(
    revisit_model,
    out_file=None,
    feature_names=revisit_features,
    class_names=['No Revisit', 'Revisit'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("topic_revisit_decision_tree", format="png", view=True)

# === 6. Save the revisit model ===
with open("topic_revisit_model.pkl", "wb") as f:
    pickle.dump(revisit_model, f)

with open("topic_revisit_encoders.pkl", "wb") as f:
    pickle.dump({
        'features': revisit_features,
        'target': revisit_target
    }, f)

print("Topic Revisit Model saved successfully!")
print("Decision tree visualization saved as 'topic_revisit_decision_tree.png'")