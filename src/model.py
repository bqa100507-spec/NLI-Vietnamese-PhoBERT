from transformers import AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def get_model(model, num_labels=3):
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels = num_labels
    )
    return model