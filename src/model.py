from transformers import AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv()

def get_model(model, num_labels=3):
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels = num_labels
    )
    return model