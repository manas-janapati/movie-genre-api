from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

app = FastAPI()

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./movie_genre_model")
tokenizer = DistilBertTokenizer.from_pretrained("./movie_genre_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load MultiLabelBinarizer
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Request body schema
class MovieDescription(BaseModel):
    description: str

# Prediction endpoint
@app.post("/predict")
async def predict_genre(movie: MovieDescription):
    try:
        # Tokenize input
        encodings = tokenizer(
            movie.description,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy() > 0.5
        predicted_genres = mlb.inverse_transform(preds)[0]

        return {"genres": predicted_genres if predicted_genres else ["Unknown"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/")
async def root():
    return {"message": "Movie Genre Prediction API"}