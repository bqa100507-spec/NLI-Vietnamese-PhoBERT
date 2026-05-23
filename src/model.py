from transformers import AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv()

def get_model(model, num_labels=3, freeze_layers=8):
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels = num_labels
    )
    for name, param in model.named_parameters():
        if "roberta.encoder.layer." in name:
            if int(name.split(".")[3]) < freeze_layers:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M out of {total_params/1e6:.2f}M")

    return model