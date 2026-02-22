from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from torchvision.transforms import transforms
from src.inference.utils import load_model
import time, logging
logging.basicConfig(level=logging.INFO)
    

app = FastAPI()
model = load_model()  # Loads mlflow model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        probs = torch.softmax(model(img), 1)
        pred = "dog" if probs[0][1] > 0.5 else "cat"
    return {"prediction": pred, "probs": probs.tolist()[0]}

    start = time.time()
    logging.info(f"Request: {file.filename}")
    # ... prediction
    latency = time.time() - start
    logging.info(f"Pred: {pred}, Latency: {latency}s")
    return {...}