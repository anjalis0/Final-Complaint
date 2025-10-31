import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from custom_model import vector_tfidf_model, parse_sentiment_file, parse_priority_file

logging.basicConfig(level=logging.INFO)


class PredictRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading models...")
    vector_tfidf_model('sentiment_vector_tfidf.pkl', ['TrainSentiment.txt'], parse_sentiment_file)

    logging.info("Sentiment model loaded.")
    vector_tfidf_model('priority_vector_tfidf.pkl', ['TrainPriority.txt'], parse_priority_file)

    logging.info("ML API is up and running ...")

    yield

    logging.info("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def read_root():
    return {"message": "pong!"}


@app.post("/predict")
async def predict(request: PredictRequest):
    sentiment_vector = vector_tfidf_model('sentiment_vector_tfidf.pkl', ['TrainSentiment.txt'], parse_sentiment_file)
    priority_vector = vector_tfidf_model('priority_vector_tfidf.pkl', ['TrainPriority.txt'], parse_priority_file)
    sentiment = sentiment_vector.predict(request.text)
    priority = priority_vector.predict(request.text)

    if priority == 'high':
        priority = "5"
    elif priority == 'medium':
        priority = "3"
    else:
        priority = "1"

    return json.dumps({"sentiment": sentiment, "priority": priority})

