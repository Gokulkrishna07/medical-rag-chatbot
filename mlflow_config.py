import mlflow

try:
    mlflow.set_tracking_uri("http://103.49.125.28:8501/mlflow/")
    mlflow.set_experiment("MedRAG-Chatbot")
except Exception as e:
    print(f"[MLflow Config] Warning: Could not initialize MLflow — {e}")