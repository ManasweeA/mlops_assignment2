import mlflow.pytorch
def load_model():
    return mlflow.pytorch.load_model("models:/<run_id>/model")
