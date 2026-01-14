import gradio as gr
import joblib
import pandas as pd
import re
import nltk
import numpy as np
import traceback
import warnings
import os

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

# --- 2. CONFIGURATION ---
# specific paths for preprocessors
VEC_TFIDF = "preprocessor/tfidf_vectorizer.joblib"
VEC_COUNT = "preprocessor/count_vectorizer.joblib"
RED_SVD   = "preprocessor/truncated_svd.joblib"

# Map each model to its specific file paths
MODEL_CONFIG = {
    "XGBoost (BoW)": {
        "model_path": "models/bow_models_without_pca/xgboost_model.joblib",
        "vec_path": VEC_COUNT,
        "red_path": None,
        "dense_required": False
    },
    "LightGBM (BoW)": {
        "model_path": "models/bow_models_without_pca/lightgbm_model.joblib",
        "vec_path": VEC_COUNT,
        "red_path": None,
        "dense_required": False
    },
    "Random Forest (BoW)": {
        "model_path": "models/bow_models_without_pca/random_forest_model.joblib",
        "vec_path": VEC_COUNT,
        "red_path": None,
        "dense_required": False
    },
    "Linear SVM (TF-IDF + SVD)": {
        "model_path": "models/tfidf_models_with_truncatedSVD/linear_svm_model.joblib",
        "vec_path": VEC_TFIDF,
        "red_path": RED_SVD,
        "dense_required": False
    },
    "Logistic Regression (TF-IDF + SVD)": {
        "model_path": "models/tfidf_models_with_truncatedSVD/logistic_regression_model.joblib",
        "vec_path": VEC_TFIDF,
        "red_path": RED_SVD,
        "dense_required": False
    }
}

# --- 3. TEXT PREPROCESSING ---
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

# --- 4. LAZY LOADING RESOURCES ---
resource_cache = {}

def get_resource(path):
    """Generic loader that handles both Windows/Linux paths safely"""
    if not path: return None
    
    full_path = os.path.normpath(path)
    
    if full_path in resource_cache:
        return resource_cache[full_path]
    
    if not os.path.exists(full_path):
        print(f"⚠️ File not found: {full_path}")
        return None
    
    print(f"⏳ Loading {full_path}...")
    try:
        obj = joblib.load(full_path)
        resource_cache[full_path] = obj
        print(f"✅ Loaded {full_path}")
        return obj
    except Exception as e:
        print(f"❌ Error loading {full_path}: {e}")
        return None

# --- 5. HELPER: SOFTMAX ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- 6. PREDICTION FUNCTION ---
def predict(text, model_choice):
    if not text: 
        return "Please enter text", {}, []
    
    if model_choice not in MODEL_CONFIG:
        return "Invalid Model Selected", {}, []
    
    config = MODEL_CONFIG[model_choice]
    
    # A. Load Vectorizer
    vectorizer = get_resource(config["vec_path"])
    if vectorizer is None:
        return f"Error: Vectorizer missing at {config['vec_path']}", {}, []
    
    # B. Load Reducer
    reducer = None
    if config["red_path"]:
        reducer = get_resource(config["red_path"])
        if reducer is None:
            return f"Error: Reducer missing at {config['red_path']}", {}, []
            
    # C. Load Model
    model = get_resource(config["model_path"])
    if model is None:
        return f"Error: Model missing at {config['model_path']}", {}, []

    try:
        # --- PIPELINE EXECUTION ---
        processed_text = khmer_tokenize(text)
        
        # 1. Vectorize
        vectors = vectorizer.transform([processed_text])
        
        # ⚠️ CRITICAL FIX: Convert Integer (BoW) to Float32 for LightGBM/XGBoost
        vectors = vectors.astype(np.float32)
        
        # 2. Dense Conversion (Only for PCA)
        if config["dense_required"]:
            vectors = vectors.toarray()
            
        # 3. Reduce (SVD/PCA)
        vectors_final = vectors
        if reducer:
            vectors_final = reducer.transform(vectors)
            # Ensure reduced vectors are also float32 (just in case)
            vectors_final = vectors_final.astype(np.float32)
        
        # --- KEYWORD EXTRACTION ---
        keywords = []
        try:
            feature_array = np.array(vectorizer.get_feature_names_out())
            
            # Check keywords using the sparse vector
            if config["dense_required"]:
                 raw_vector_check = vectorizer.transform([processed_text])
            else:
                 raw_vector_check = vectors
                 
            tfidf_sorting = np.argsort(raw_vector_check.toarray()).flatten()[::-1]
            top_n = 10
            for idx in tfidf_sorting[:top_n]:
                if raw_vector_check[0, idx] > 0:
                    keywords.append(feature_array[idx])
        except:
            keywords = ["Keywords N/A"]

        # --- PREDICTION ---
        confidences = {}
        top_label = ""
        
        # Strategy 1: Probabilities (Trees, LogReg)
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(vectors_final)[0]
                for i in range(len(LABELS)):
                    if i < len(probas):
                        confidences[LABELS[i]] = float(probas[i])
                top_label = max(confidences, key=confidences.get)
            except Exception as e: 
                print(f"predict_proba failed: {e}")

        # Strategy 2: Decision Function (SVM fallback)
        if not confidences and hasattr(model, "decision_function"):
            try:
                raw_scores = model.decision_function(vectors_final)[0]
                probas = softmax(raw_scores)
                for i in range(len(LABELS)):
                    if i < len(probas):
                        confidences[LABELS[i]] = float(probas[i])
                top_label = max(confidences, key=confidences.get)
            except Exception as e:
                print(f"decision_function failed: {e}")

        # Strategy 3: Hard Fallback (Last resort)
        if not confidences:
            try:
                raw_pred = model.predict(vectors_final)[0]
                if isinstance(raw_pred, (int, np.integer, float, np.floating)):
                     pred_idx = int(raw_pred)
                     top_label = LABELS[pred_idx]
                else:
                     top_label = str(raw_pred)
                confidences = {top_label: 1.0}
            except Exception as e:
                return f"Prediction Failed: {str(e)}", {}, []

        return top_label, confidences, keywords
            
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", {}, []

# --- 7. LAUNCH ---
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter Khmer news text here...", label="Input Text"),
        gr.Dropdown(choices=list(MODEL_CONFIG.keys()), value="XGBoost", label="Select Model")
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
    app.launch()