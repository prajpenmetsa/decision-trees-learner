import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
import html

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
    "Visual Navigator": "Use simple diagrams, illustrations per subtopic. Don’t repeat old material.",
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

# --- 2. Data Definition with Strategy Labels as Target (based on refined rules) ---
# Each row now corresponds precisely to one of your new, mutually exclusive rules.
initial_data_raw = []

# Revisiting Module = True (R-rules: Learner Revisiting)
# attention_span < 9
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 5, 'format_preference': 'text', 'intervention_needed': "Short-Text Bouncer"}, # R1
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 7, 'format_preference': 'image', 'intervention_needed': "Visual Burst Learner"}, # R2
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Video Refresher"}, # R3
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 4, 'format_preference': 'text', 'intervention_needed': "Mismatch Text Splitter"}, # R7
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 6, 'format_preference': 'image', 'intervention_needed': "Mismatch Visual Cutter"}, # R8
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Mismatch Video Snacker"}, # R9
])

# attention_span >= 11
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 15, 'format_preference': 'text', 'intervention_needed': "Focused Text Learner"}, # R4
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 20, 'format_preference': 'image', 'intervention_needed': "Visual Concentrator"}, # R5
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 25, 'format_preference': 'video', 'intervention_needed': "Video Explainer"}, # R6
])

# attention_span >= 9 AND attention_span < 11
initial_data_raw.extend([
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 9, 'format_preference': 'text', 'intervention_needed': "Adaptive Text Coach"}, # R10 (short_chunks, text)
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 10, 'format_preference': 'text', 'intervention_needed': "Adaptive Text Coach"}, # R10 (long_sessions, text)
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 9.5, 'format_preference': 'image', 'intervention_needed': "Adaptive Text Coach"}, # R10 (short_chunks, image)
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 10.5, 'format_preference': 'video', 'intervention_needed': "Adaptive Text Coach"}, # R10 (short_chunks, video)
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 9.2, 'format_preference': 'image', 'intervention_needed': "Adaptive Visual Guide"}, # R11 (long_sessions, image)
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 10.8, 'format_preference': 'video', 'intervention_needed': "Adaptive Video Segments"}, # R12 (long_sessions, video)
])


# Revisiting Module = False (F-rules: Learner Progressing)
# attention_span < 9
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 5, 'format_preference': 'text', 'intervention_needed': "Chunked Forward Mover"}, # F1
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 7, 'format_preference': 'image', 'intervention_needed': "Visual Navigator"}, # F2
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Video Streamline Learner"}, # F3
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 4, 'format_preference': 'text', 'intervention_needed': "Mismatch Text Forwarder"}, # F7
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 6, 'format_preference': 'image', 'intervention_needed': "Mismatch Visual Forwarder"}, # F8
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 8, 'format_preference': 'video', 'intervention_needed': "Mismatch Video Forwarder"}, # F9
])

# attention_span >= 11
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 15, 'format_preference': 'text', 'intervention_needed': "Text Mastery Learner"}, # F4
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 20, 'format_preference': 'image', 'intervention_needed': "Visual Immersionist"}, # F5
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 25, 'format_preference': 'video', 'intervention_needed': "Video Focused Learner"}, # F6
])

# attention_span >= 9 AND attention_span < 11
initial_data_raw.extend([
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 9, 'format_preference': 'text', 'intervention_needed': "Forward Text Balance"}, # F10 (short_chunks, text)
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 10, 'format_preference': 'text', 'intervention_needed': "Forward Text Balance"}, # F10 (long_sessions, text)
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 9.5, 'format_preference': 'image', 'intervention_needed': "Forward Text Balance"}, # F10 (short_chunks, image)
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 10.5, 'format_preference': 'video', 'intervention_needed': "Forward Text Balance"}, # F10 (short_chunks, video)
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 9.2, 'format_preference': 'image', 'intervention_needed': "Forward Visual Balance"}, # F11 (long_sessions, image)
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 10.8, 'format_preference': 'video', 'intervention_needed': "Forward Video Balance"}, # F12 (long_sessions, video)
])


# --- Simulated New User Data Flow (split into batches for demonstration) ---
# These now align with the new, mutually exclusive rules.
simulated_new_data_raw = [
    # Batch 1: Early observations
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 9.3, 'format_preference': 'text', 'intervention_needed': "Adaptive Text Coach"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 10.2, 'format_preference': 'image', 'intervention_needed': "Forward Text Balance"}, # This goes to F10 based on (short_chunks OR text)

    # Batch 2: More observations
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 8.7, 'format_preference': 'video', 'intervention_needed': "Video Refresher"},
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 10.9, 'format_preference': 'video', 'intervention_needed': "Forward Video Balance"},

    # Batch 3: Even more observations
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 11.5, 'format_preference': 'image', 'intervention_needed': "Visual Concentrator"},
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 4.5, 'format_preference': 'text', 'intervention_needed': "Chunked Forward Mover"},
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 9.8, 'format_preference': 'image', 'intervention_needed': "Adaptive Visual Guide"},
]

# --- Encoders (fit on all possible values) ---
all_session_prefs = ['short_chunks', 'long_sessions']
all_format_prefs = ['text', 'image', 'video']
all_strategy_labels = list(strategy_label_to_llm_prompt.keys())

le_session = LabelEncoder().fit(all_session_prefs)
le_format = LabelEncoder().fit(all_format_prefs)
le_strategy_label = LabelEncoder().fit(all_strategy_labels)

def encode_dataframe(df_to_encode, le_session, le_format, le_strategy_label):
    df_encoded = df_to_encode.copy()
    df_encoded['revisiting_module_encoded'] = df_encoded['revisiting_module'].astype(int)
    df_encoded['session_preference_encoded'] = le_session.transform(df_encoded['session_preference'])
    df_encoded['format_preference_encoded'] = le_format.transform(df_encoded['format_preference'])
    df_encoded['intervention_needed_encoded'] = le_strategy_label.transform(df_encoded['intervention_needed'])
    return df_encoded

# --- Function to generate user categories (descriptive strings) ---
def get_user_category_description(row):
    revisiting = "Revisiting" if row['revisiting_module'] else "Progressing"
    session_pref = row['session_preference'].replace('_', ' ').title()
    format_pref = row['format_preference'].title()

    attention_span = row['attention_span']
    if attention_span < 9:
        att_cat = "Low (<9 min)"
    elif attention_span >= 11:
        att_cat = "High (>=11 min)"
    else: # attention_span >= 9 AND attention_span < 11
        att_cat = "Intermediate (9-11 min)"

    return f"{revisiting} | Session: {session_pref} | Attention: {att_cat} | Format: {format_pref}"

# Function to run predictions and generate output map for a given model and test data
def generate_output_map(model, test_df_raw, le_session, le_format, le_strategy_label, strategy_label_to_llm_prompt, title=""):
    test_df_encoded = encode_dataframe(test_df_raw, le_session, le_format, le_strategy_label)
    X_test_final = test_df_encoded[['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']]
    y_test_actual_encoded = test_df_encoded['intervention_needed_encoded']

    y_test_predicted_encoded = model.predict(X_test_final)

    y_test_predicted_strategy_labels = le_strategy_label.inverse_transform(y_test_predicted_encoded)
    y_test_actual_strategy_labels = le_strategy_label.inverse_transform(y_test_actual_encoded)

    y_test_predicted_llm_prompts = [strategy_label_to_llm_prompt[label] for label in y_test_predicted_strategy_labels]
    y_test_actual_llm_prompts = [strategy_label_to_llm_prompt[label] for label in y_test_actual_strategy_labels]

    output_map = pd.DataFrame({
        'User Category': test_df_raw.apply(get_user_category_description, axis=1),
        'Actual Strategy Label': y_test_actual_strategy_labels,
        'Predicted Strategy Label': y_test_predicted_strategy_labels,
        'Actual LLM Prompt Summary': y_test_actual_llm_prompts,
        'Predicted LLM Prompt Summary': y_test_predicted_llm_prompts
    })
    print(f"\n--- {title} Predictions & User Category Map ---")
    print(output_map.to_string())

# --- Test Data to Observe Changes ---
# Use a consistent test set to see how predictions change with model adaptation
# These test cases cover a range of scenarios based on the new rules.
evaluation_test_data_raw = pd.DataFrame([
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 7.5, 'format_preference': 'text', 'intervention_needed': "Short-Text Bouncer"}, # R1
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 12.0, 'format_preference': 'image', 'intervention_needed': "Visual Immersionist"}, # F5
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 9.8, 'format_preference': 'video', 'intervention_needed': "Adaptive Video Segments"}, # R12
    {'revisiting_module': False, 'session_preference': 'short_chunks', 'attention_span': 10.1, 'format_preference': 'text', 'intervention_needed': "Forward Text Balance"}, # F10
    {'revisiting_module': True, 'session_preference': 'short_chunks', 'attention_span': 8.9, 'format_preference': 'image', 'intervention_needed': "Visual Burst Learner"}, # R2 (attention_span < 9, near boundary)
    {'revisiting_module': False, 'session_preference': 'long_sessions', 'attention_span': 8.9, 'format_preference': 'text', 'intervention_needed': "Mismatch Text Forwarder"}, # F7 (attention_span < 9, near boundary)
    {'revisiting_module': True, 'session_preference': 'long_sessions', 'attention_span': 10.9, 'format_preference': 'image', 'intervention_needed': "Adaptive Visual Guide"}, # R11 (attention_span < 11, near boundary)
])

# Escaped strategy labels for Graphviz (for class_names in the tree image)
escaped_strategy_labels = [html.escape(label) for label in le_strategy_label.classes_]

print("--- Sequential Model Adaptation Demonstration with Mutually Exclusive Rules ---")
print("Observe how the tree structure and predictions change as more data is added.")
print("New tree visualizations will be saved as .png files for each stage.")

# =========================================================
## Stage 1: Initial Tree (Only Predefined Rules - Mutually Exclusive)
# =========================================================
current_training_df = pd.DataFrame(initial_data_raw)
current_training_df_encoded = encode_dataframe(current_training_df, le_session, le_format, le_strategy_label)

X_train_s1 = current_training_df_encoded[['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']]
y_train_s1 = current_training_df_encoded['intervention_needed_encoded']

model_s1 = DecisionTreeClassifier(random_state=42, max_depth=None)
model_s1.fit(X_train_s1, y_train_s1)

print("\n\n--- Stage 1: Initial Model Trained (Mutually Exclusive Rules Only) ---")
dot_data_s1 = export_graphviz(
    model_s1, out_file=None, feature_names=X_train_s1.columns,
    class_names=escaped_strategy_labels, filled=True, rounded=True, special_characters=True
)
graph_s1 = graphviz.Source(dot_data_s1)
graph_s1.render("stage1_initial_tree_mutually_exclusive", format="png", view=False)
print("Stage 1 tree visualization saved as stage1_initial_tree_mutually_exclusive.png")

generate_output_map(model_s1, evaluation_test_data_raw, le_session, le_format, le_strategy_label, strategy_label_to_llm_prompt, "Stage 1 Model (Mutually Exclusive Rules)")

# =========================================================
## Stage 2: Adapt with Batch 1 of New Data
# =========================================================
new_data_batch1 = pd.DataFrame(simulated_new_data_raw[0:2]) # First 2 samples
current_training_df = pd.concat([current_training_df, new_data_batch1], ignore_index=True)
current_training_df_encoded = encode_dataframe(current_training_df, le_session, le_format, le_strategy_label)

X_train_s2 = current_training_df_encoded[['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']]
y_train_s2 = current_training_df_encoded['intervention_needed_encoded']

model_s2 = DecisionTreeClassifier(random_state=42, max_depth=None)
model_s2.fit(X_train_s2, y_train_s2)

print("\n\n--- Stage 2: Model Adapted with Batch 1 Data (Mutually Exclusive Rules) ---")
dot_data_s2 = export_graphviz(
    model_s2, out_file=None, feature_names=X_train_s2.columns,
    class_names=escaped_strategy_labels, filled=True, rounded=True, special_characters=True
)
graph_s2 = graphviz.Source(dot_data_s2)
graph_s2.render("stage2_adaptive_tree_batch1_mutually_exclusive", format="png", view=False)
print("Stage 2 tree visualization saved as stage2_adaptive_tree_batch1_mutually_exclusive.png")

generate_output_map(model_s2, evaluation_test_data_raw, le_session, le_format, le_strategy_label, strategy_label_to_llm_prompt, "Stage 2 Model (After Batch 1, Mutually Exclusive Rules)")


# =========================================================
## Stage 3: Adapt with Batch 2 of New Data
# =========================================================
new_data_batch2 = pd.DataFrame(simulated_new_data_raw[2:4]) # Next 2 samples
current_training_df = pd.concat([current_training_df, new_data_batch2], ignore_index=True)
current_training_df_encoded = encode_dataframe(current_training_df, le_session, le_format, le_strategy_label)

X_train_s3 = current_training_df_encoded[['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']]
y_train_s3 = current_training_df_encoded['intervention_needed_encoded']

model_s3 = DecisionTreeClassifier(random_state=42, max_depth=None)
model_s3.fit(X_train_s3, y_train_s3)

print("\n\n--- Stage 3: Model Adapted with Batch 2 Data (Mutually Exclusive Rules) ---")
dot_data_s3 = export_graphviz(
    model_s3, out_file=None, feature_names=X_train_s3.columns,
    class_names=escaped_strategy_labels, filled=True, rounded=True, special_characters=True
)
graph_s3 = graphviz.Source(dot_data_s3)
graph_s3.render("stage3_adaptive_tree_batch2_mutually_exclusive", format="png", view=False)
print("Stage 3 tree visualization saved as stage3_adaptive_tree_batch2_mutually_exclusive.png")

generate_output_map(model_s3, evaluation_test_data_raw, le_session, le_format, le_strategy_label, strategy_label_to_llm_prompt, "Stage 3 Model (After Batch 2, Mutually Exclusive Rules)")


# =========================================================
## Stage 4: Adapt with Batch 3 (Remaining) of New Data
# = ========================================================
new_data_batch3 = pd.DataFrame(simulated_new_data_raw[4:]) # Remaining samples
current_training_df = pd.concat([current_training_df, new_data_batch3], ignore_index=True)
current_training_df_encoded = encode_dataframe(current_training_df, le_session, le_format, le_strategy_label)

X_train_s4 = current_training_df_encoded[['revisiting_module_encoded', 'session_preference_encoded', 'attention_span', 'format_preference_encoded']]
y_train_s4 = current_training_df_encoded['intervention_needed_encoded']

model_s4 = DecisionTreeClassifier(random_state=42, max_depth=None)
model_s4.fit(X_train_s4, y_train_s4)

print("\n\n--- Stage 4: Model Adapted with Batch 3 Data (All New Data, Mutually Exclusive Rules) ---")
dot_data_s4 = export_graphviz(
    model_s4, out_file=None, feature_names=X_train_s4.columns,
    class_names=escaped_strategy_labels, filled=True, rounded=True, special_characters=True
)
graph_s4 = graphviz.Source(dot_data_s4)
graph_s4.render("stage4_adaptive_tree_final_mutually_exclusive", format="png", view=True) # View the final tree
print("Stage 4 tree visualization saved as stage4_adaptive_tree_final_mutually_exclusive.png")

generate_output_map(model_s4, evaluation_test_data_raw, le_session, le_format, le_strategy_label, strategy_label_to_llm_prompt, "Stage 4 Model (Final, Mutually Exclusive Rules)")