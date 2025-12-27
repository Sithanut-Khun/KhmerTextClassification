from huggingface_hub import HfApi, create_repo, upload_file
import joblib
import os

# Login once (it will open a browser to get a token)
from huggingface_hub import login
login()  


repo_id = "Thanut003/khmer-news-classifier"  
create_repo(repo_id=repo_id, repo_type="model", private=False)  # Set private=True if needed

# Upload your joblib files
api = HfApi()

print("Starting model upload to Hugging Face Hub...")
print(f"Repository: {repo_id}\n")

# If you saved separate components (e.g., tfidf, svd, clf)
# Get the project root directory (parent of src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models", "trained")

# Upload all models
upload_file(path_or_fileobj=os.path.join(models_dir, "lightgbm_model.joblib"), path_in_repo="lightgbm_model.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "tfidf_vectorizer.joblib"), path_in_repo="tfidf_vectorizer.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "truncated_svd.joblib"), path_in_repo="truncated_svd.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "logistic_regression_model.joblib"), path_in_repo="logistic_regression_model.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "linear_svm_model.joblib"), path_in_repo="linear_svm_model.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "random_forest_model.joblib"), path_in_repo="random_forest_model.joblib", repo_id=repo_id, repo_type="model")
upload_file(path_or_fileobj=os.path.join(models_dir, "xgboost_model.joblib"), path_in_repo="xgboost_model.joblib", repo_id=repo_id, repo_type="model")

