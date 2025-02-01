from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.bert_classifier import ModelTrainer
import torch
import os

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Define paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pth')

# Initialize model
model_trainer = ModelTrainer(num_classes=4)

# Load the trained model
try:
    model_trainer.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

label_map = {
    0: 'Electronics',
    1: 'Household',
    2: 'Books',
    3: 'Clothing & Accessories'
}

@app.post("/predict")
async def predict(text_input: TextInput):
    try:
        prediction = model_trainer.predict(text_input.text)
        category = label_map[prediction]
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "E-commerce Text Classification API"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(model_trainer.device)
    }
