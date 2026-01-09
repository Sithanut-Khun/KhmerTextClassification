import gradio as gr
import joblib
import pandas as pd
import re
import nltk
import numpy as np
import traceback
import warnings

# --- 1. SETUP ---
warnings.filterwarnings("ignore")

from khmernltk import word_tokenize

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))

# LABELS
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

# --- HELPER: SOFTMAX ---
# Converts raw distance scores (e.g., -1.5, 2.3) into probabilities (e.g., 0.1, 0.8)
def softmax(x):
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / e_x.sum()

# --- 2. LAZY LOADING ---
vectorizer = None
svd = None
models_cache = {} 

model_files = {
    "XGBoost": "xgboost_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
    "Linear SVM": "linear_svm_model.joblib"
}

def load_vectorizers():
    global vectorizer, svd
    if vectorizer is None:
        try:
            vectorizer = joblib.load("tfidf_vectorizer.joblib")
            svd = joblib.load("truncated_svd.joblib")
        except Exception as e:
            print(f"Error loading vectorizers: {e}")
            return False
    return True

def get_model(name):
    if name in models_cache:
        return models_cache[name]
    try:
        filename = model_files.get(name)
        if not filename: return None
        loaded_model = joblib.load(filename)
        models_cache[name] = loaded_model 
        return loaded_model
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

# --- 3. PREDICTION FUNCTION ---
def predict(text, model_name):
    if not text: 
        return "Please enter text", {}, []
    
    if not load_vectorizers():
        return "System Error: Vectorizers missing", {}, []

    current_model = get_model(model_name)
    if current_model is None:
        return f"Error: Could not load {model_name}", {}, []

    try:
        processed = khmer_tokenize(text)
        vectors = vectorizer.transform([processed])
        vectors_reduced = svd.transform(vectors)
        
        # --- Keyword Extraction ---
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(vectors.toarray()).flatten()[::-1]
        
        top_n = 10
        keywords = []
        for idx in tfidf_sorting[:top_n]:
            if vectors[0, idx] > 0:
                keywords.append(feature_array[idx])

        # --- Prediction Logic ---
        confidences = {}
        top_label = ""
        
        # STRATEGY 1: NATIVE PROBABILITIES (XGBoost, RF, LogReg)
        if hasattr(current_model, "predict_proba"):
            try:
                probas = current_model.predict_proba(vectors_reduced)[0]
                for i in range(len(LABELS)):
                    if i < len(probas):
                        confidences[LABELS[i]] = float(probas[i])
                top_label = max(confidences, key=confidences.get)
            except:
                # Fallback if predict_proba fails
                pass

        # STRATEGY 2: DECISION FUNCTION (SVM fallback)
        # If strategy 1 didn't work, we try to use "distance" scores and convert them
        if not confidences and hasattr(current_model, "decision_function"):
            try:
                raw_scores = current_model.decision_function(vectors_reduced)[0]
                # Convert raw scores (distances) to percentages using Softmax
                probas = softmax(raw_scores)
                
                for i in range(len(LABELS)):
                    if i < len(probas):
                        confidences[LABELS[i]] = float(probas[i])
                top_label = max(confidences, key=confidences.get)
            except:
                pass

        # STRATEGY 3: HARD FALLBACK (If everything else fails)
        if not confidences:
            raw_pred = current_model.predict(vectors_reduced)[0]
            if isinstance(raw_pred, (int, np.integer, float, np.floating)):
                 pred_idx = int(raw_pred)
                 top_label = LABELS[pred_idx]
            else:
                 top_label = str(raw_pred)
            confidences = {top_label: 1.0}

        return top_label, confidences, keywords
            
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", {}, []

# --- 4. LAUNCH ---
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter Khmer news text here...", label="Input Text"), 
        gr.Dropdown(choices=list(model_files.keys()), value="XGBoost", label="Select Model")
    ],
    outputs=[
        gr.Label(label="Top Prediction"), 
        gr.Label(num_top_classes=8, label="Class Probabilities"), 
        gr.JSON(label="Top Keywords")
    ],
    title="Khmer News Classifier",
    description="Classify Khmer text into 8 categories."
)

if __name__ == "__main__":
    demo.launch()
