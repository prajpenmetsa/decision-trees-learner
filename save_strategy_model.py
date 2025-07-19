import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html
import pickle

# === 1. Define strategy label to LLM prompt mapping ===
strategy_label_to_llm_prompt = {
    "Short-Text Bouncer": "Break down module in short text blocks (≤5 min). Emphasize simplicity and repetition.",
    "Visual Burst Learner": "Deliver 1-image-per-concept slides or flowcharts per idea.",
    "Video Refresher": "Provide 2–3 min videos per topic with transcripts.",
    "Focused Text Learner": "Deliver entire module in 20–25 min guided text tutorials.",
    "Visual Concentrator": "Teach full module using labeled diagrams & conceptual maps.",
    "Video Explainer": "Deliver 10–12 min instructional videos with mid-video check questions.",
    "Mismatch Text Splitter": "Mismatch: Split content into 5–6 min text pieces even though long format is preferred.",
    "Mismatch Visual Cutter": "Split visual concepts into sequential illustrations.",
    "Mismatch Video Snacker": "Deliver microlearning video bites with concept labeling.",
    "Adaptive Text Coach": "Split topics into ~7 min text sections with structured guidance, adapting to preference.",
    "Adaptive Visual Guide": "Adaptive visual delivery in sequential chunks for intermediate attention.",
    "Adaptive Video Segments": "Adaptive video delivery in concise segments for intermediate attention.",
    "Chunked Forward Mover": "Continue with next module in short, focused text blocks. Emphasize key takeaways.",
    "Visual Navigator": "Use simple diagrams, illustrations per subtopic. Don't repeat old material.",
    "Video Streamline Learner": "Deliver next module in 2–4 min videos per concept. Avoid repetition.",
    "Text Mastery Learner": "Deliver module as in-depth textual explanations with summaries.",
    "Visual Immersionist": "Use concept maps and complete visual walkthroughs for new content.",
    "Video Focused Learner": "Long-format instructional videos for new content. Mid-video Q&A checkpoints.",
    "Mismatch Text Forwarder": "Break forward content into shorter blocks to match learner attention.",
    "Mismatch Visual Forwarder": "Slice diagrams by concept and arrange sequentially.",
    "Mismatch Video Forwarder": "Convert forward module into small standalone videos.",
    "Forward Text Balance": "Deliver next module in hybrid text chunks (7–9 mins). Add reflection points, adapting to preference.",
    "Forward Visual Balance": "Forward visual content in adaptive segments for intermediate attention.",
    "Forward Video Balance": "Forward video content in adaptive segments for intermediate attention."
}

# === 2. Define training data based on mutually exclusive rules ===
initial_data_raw = []

# Revisiting Module = True (R-rules: Learner Revisiting)
# attention_span < 9
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 5, 'format_preference': 'text', 'intervention_needed': "Short-Text Bouncer"},
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 7, 'format_preference': 'image', 'intervention_needed': "Visual Burst Learner"},
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Video Refresher"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 4, 'format_preference': 'text', 'intervention_needed': "Mismatch Text Splitter"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 6, 'format_preference': 'image', 'intervention_needed': "Mismatch Visual Cutter"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Mismatch Video Snacker"},
])

# attention_span >= 11
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 15, 'format_preference': 'text', 'intervention_needed': "Focused Text Learner"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 20, 'format_preference': 'image', 'intervention_needed': "Visual Concentrator"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 25, 'format_preference': 'video', 'intervention_needed': "Video Explainer"},
])

# attention_span >= 9 AND attention_span < 11
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 9, 'format_preference': 'text', 'intervention_needed': "Adaptive Text Coach"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 10, 'format_preference': 'text', 'intervention_needed': "Adaptive Text Coach"},
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 9.5, 'format_preference': 'image', 'intervention_needed': "Adaptive Text Coach"},
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 10.5, 'format_preference': 'video', 'intervention_needed': "Adaptive Text Coach"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 9.2, 'format_preference': 'image', 'intervention_needed': "Adaptive Visual Guide"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 10.8, 'format_preference': 'video', 'intervention_needed': "Adaptive Video Segments"},
])

# Revisiting Module = False (F-rules: Learner Progressing)
# attention_span < 9
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 5, 'format_preference': 'text', 'intervention_needed': "Chunked Forward Mover"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 7, 'format_preference': 'image', 'intervention_needed': "Visual Navigator"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Video Streamline Learner"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 4, 'format_preference': 'text', 'intervention_needed': "Mismatch Text Forwarder"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 6, 'format_preference': 'image', 'intervention_needed': "Mismatch Visual Forwarder"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Mismatch Video Forwarder"},
])

# attention_span >= 11
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 15, 'format_preference': 'text', 'intervention_needed': "Text Mastery Learner"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 20, 'format_preference': 'image', 'intervention_needed': "Visual Immersionist"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 25, 'format_preference': 'video', 'intervention_needed': "Video Focused Learner"},
])

# attention_span >= 9 AND attention_span < 11
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 9, 'format_preference': 'text', 'intervention_needed': "Forward Text Balance"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 10, 'format_preference': 'text', 'intervention_needed': "Forward Text Balance"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 9.5, 'format_preference': 'image', 'intervention_needed': "Forward Text Balance"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 10.5, 'format_preference': 'video', 'intervention_needed': "Forward Text Balance"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 9.2, 'format_preference': 'image', 'intervention_needed': "Forward Visual Balance"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 10.8, 'format_preference': 'video', 'intervention_needed': "Forward Video Balance"},
])

# === 3. Prepare data for training ===
df = pd.DataFrame(initial_data_raw)

# === 4. Encode categorical variables ===
all_session_prefs = ['short_chunks', 'long_sessions']
all_format_prefs = ['text', 'image', 'video']
all_strategy_labels = list(strategy_label_to_llm_prompt.keys())

le_session = LabelEncoder().fit(all_session_prefs)
le_format = LabelEncoder().fit(all_format_prefs)
le_strategy_label = LabelEncoder().fit(all_strategy_labels)

df['revisiting_module_encoded'] = df['revisiting_module'].astype(int)
df['session_preference_encoded'] = le_session.transform(df['session_preference'])
df['format_preference_encoded'] = le_format.transform(df['format_preference'])
df['intervention_needed_encoded'] = le_strategy_label.transform(df['intervention_needed'])

# === 5. Prepare features and target ===
features = ['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']
X = df[features]
y = df['intervention_needed_encoded']

# === 6. Train the decision tree model ===
strategy_model = DecisionTreeClassifier(random_state=42, max_depth=None)
strategy_model.fit(X, y)

# === 7. Visualize using Graphviz ===
escaped_strategy_labels = [html.escape(label) for label in le_strategy_label.classes_]
dot_data = export_graphviz(
    strategy_model,
    out_file=None,
    feature_names=features,
    class_names=escaped_strategy_labels,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("strategy_decision_tree", format="png", view=True)

# === 8. Save model and encoders ===
with open("strategy_model.pkl", "wb") as f:
    pickle.dump(strategy_model, f)

with open("strategy_encoders.pkl", "wb") as f:
    pickle.dump({
        'session_encoder': le_session,
        'format_encoder': le_format,
        'strategy_encoder': le_strategy_label,
        'strategy_prompts': strategy_label_to_llm_prompt,
        'features': features
    }, f)

print("Strategy Model and encoders saved successfully!")
print("Decision tree visualization saved as 'strategy_decision_tree.png'")