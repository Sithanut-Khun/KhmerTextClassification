# 🇰🇭 Khmer News Text Classification AI

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-cyan)
![Hugging Face](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-yellow)

An end-to-end Full Stack Machine Learning application capable of classifying Khmer language news articles into 8 distinct categories (Culture, Economics, Politics, etc.) with high accuracy. The project features a modern, dual-language (English/Khmer) user interface.

## 🚀 Live Demo
- **Frontend (UI):** [Click here to view the Website](https://YOUR_USERNAME.github.io/KhmerTextClassification) *(Replace with your GitHub Pages link)*
- **Backend (API):** [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME)

---

## ✨ Key Features

* **Multi-Model Support:** Users can switch between 5 different trained models in real-time:
    * XGBoost (Best Accuracy)
    * LightGBM (Fastest Inference)
    * Random Forest
    * Logistic Regression
    * Linear SVM
* **Explainability:** The AI extracts and highlights the top **10 keywords** that influenced the decision using TF-IDF weighting.
* **Dual Language UI:** Fully localized interface supporting both **English** and **Khmer** languages.
* **Interactive Visualization:** Real-time confidence bars showing the probability distribution across all categories.
* **Modern Architecture:** "Serverless" style architecture separating the heavy ML inference (Hugging Face) from the lightweight UI (GitHub Pages).

---

## 🛠️ Tech Stack

### **Data Science & NLP**
* **Preprocessing:** `khmer-nltk` for word segmentation, Custom Regex cleaning.
* **Feature Engineering:** TF-IDF Vectorization (8000 features) → TruncatedSVD (Dimensionality Reduction).
* **Models:** Scikit-learn, XGBoost, LightGBM.

### **Backend (Hugging Face)**
* **Framework:** Python `Gradio` (used as an API).
* **Deployment:** Hugging Face Spaces (CPU Basic).

### **Frontend (GitHub Pages)**
* **Framework:** React.js (Vite).
* **Styling:** Tailwind CSS v4.
* **API Client:** `@gradio/client` for stable connection to the Python backend.

---

## 📂 Project Structure

```text
KhmerTextClassification/
├── data/                      # Raw and Processed datasets
├── models/                    # Trained .joblib models (XGBoost, Vectorizers, SVD, etc.)
├── notebooks/                 # Jupyter Notebooks for training & analysis
├── deployment/
│   └── huggingface/           # Python code for the Backend API (app.py)
├── web-app/                   # React Frontend Application
│   ├── src/
│   │   ├── App.jsx            # Main UI Logic
│   │   └── index.css          # Tailwind Styles
│   ├── package.json
│   └── vite.config.js
└── README.md                  # Project Documentation


⚡ How to Run Locally
1. Backend (Python API)
Note: You usually don't need to run this locally if the Hugging Face space is live, but for debugging:

cd deployment/huggingface
pip install -r requirements.txt
python app.py

