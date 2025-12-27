import gradio as gr
import joblib
import pandas as pd
import re
import nltk
import numpy as np
from khmernltk import word_tokenize

# --- 1. SETUP ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))

# CRITICAL: This list MUST match the order of your LabelEncoder classes (0, 1, 2...)
LABELS = [
    'Culture', 'Economic', 'Education', 'Environment',
    'Health', 'Politics', 'Human Rights', 'Science'
]

def clean_khmer_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'[!"#$%&\'()*+,—./:;<=>?@[\]^_`{|}~។៕៖ៗ៘៙៚៛«»-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def khmer_tokenize(text):
    cleaned = clean_khmer_text(text)
    if not cleaned: return ""
    tokens = word_tokenize(cleaned)
    processed_tokens = []
    for token in tokens:
        if re.match(r'^[a-zA-Z0-9]+$', token):
            token_lower = token.lower()
            if token_lower in english_stopwords: continue
            processed_tokens.append(token_lower)
        else:
            processed_tokens.append(token)
    return " ".join(processed_tokens)

# --- 2. LOAD MODELS ---
print("Loading processors...")
try:
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    svd = joblib.load("truncated_svd.joblib")
    print("✅ Vectorizer & SVD loaded")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {e}")

models = {}
model_files = {
    "XGBoost": "xgboost_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
    "Linear SVM": "linear_svm_model.joblib"
}

for name, filename in model_files.items():
    try:
        models[name] = joblib.load(filename)
        print(f"✅ Loaded {name}")
    except:
        print(f"⚠️ Skipping {name}")

# --- 3. PREDICTION FUNCTION ---
# --- 3. PREDICTION FUNCTION ---
def predict(text, model_name):
    if not text: return "Please enter text", {}, []
    if model_name not in models: return "Model not found", {}, []
    
    try:
        # Pipeline
        processed = khmer_tokenize(text)
        vectors = vectorizer.transform([processed]) # TF-IDF Matrix (Sparse)
        vectors_reduced = svd.transform(vectors)    # SVD Matrix (Dense)
        model = models[model_name]
        
        # --- NEW: EXTRACT KEYWORDS ---
        # We look at the TF-IDF vector (before SVD) to find the strongest words
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(vectors.toarray()).flatten()[::-1]
        
        # Get top 10 words that actually have a score > 0
        top_n = 10
        keywords = []
        for idx in tfidf_sorting[:top_n]:
            if vectors[0, idx] > 0:
                keywords.append(feature_array[idx])
        
        # --- PREDICTION ---
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(vectors_reduced)[0]
            confidences = {LABELS[i]: float(probas[i]) for i in range(len(LABELS))}
            top_label = max(confidences, key=confidences.get)
        else:
            raw_pred = model.predict(vectors_reduced)[0]
            pred_idx = int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else np.argmax(raw_pred)
            top_label = LABELS[pred_idx]
            confidences = {LABELS[pred_idx]: 1.0}
            
        return top_label, confidences, keywords
        
    except Exception as e:
        return f"Error: {str(e)}", {}, []

# --- 4. LAUNCH ---
# IMPORTANT: allowed_origins="*" fixes the 405 error
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(), gr.Dropdown(choices=list(models.keys()))],
    outputs=[gr.Label(), gr.Label(), gr.JSON()]
)
demo.launch()