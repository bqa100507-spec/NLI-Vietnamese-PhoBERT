import torch
from torch.utils.data import DataLoader
import pandas as pd
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
transformers.logging.set_verbosity_error()
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from dataset import NLIDataset
from model import get_model


from dotenv import load_dotenv
import os
load_dotenv()

MODEL_NAME = "vinai/phobert-base"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 16
ACCUMULATION_STEPS = 2
MAX_LENGTH = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

splits = {'train': 'vianli_train.jsonl', 'validation': 'vianli_dev.jsonl', 'test': 'vianli_test.jsonl'}
df_train = pd.read_json("hf://datasets/uitnlp/ViANLI/" + splits["train"], lines=True)
df_valid = pd.read_json("hf://datasets/uitnlp/ViANLI/" + splits["validation"], lines=True)

if device == "cpu":
    print("Using CPU for training")
    print("Reducing the size of the training and validation sets")
    df_train = df_train.head(20)
    df_valid = df_valid.head(10)
    BATCH_SIZE = 2
    EPOCHS = 1 
    ACCUMULATION_STEPS = 1

print("train shape:", df_train.shape)
print("valid shape:", df_valid.shape)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, num_labels=len(df_train["label"].unique()))
train_dataset = NLIDataset(
    df_train,
    tokenizer,
    max_length=MAX_LENGTH
)
label_dict = np.sort(list(train_dataset.label_dict.keys()))
valid_dataset = NLIDataset(
    df_valid,
    tokenizer,
    label_list=label_dict,
    max_length=MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

model = get_model(MODEL_NAME, num_labels=len(label_dict))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_train_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(num_train_steps * 0.1),
    num_training_steps=num_train_steps
)

if device == 'cuda':
    scaler = torch.amp.GradScaler('cuda')
else:
    scaler = None

print("Starting training ...")
best_f1 = 0.0
history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
}
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss = loss / ACCUMULATION_STEPS 
            scaler.scale(loss).backward() 
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

        train_loss += loss.item() * ACCUMULATION_STEPS
        
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            if device == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}")

    if device == 'cuda':
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        with torch.inference_mode(): 
            val_progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validating")
            for batch in val_progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss/len(valid_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved best model with F1: ", best_f1)

    if device == 'cuda':
        history["train_loss"].append(train_loss/len(train_loader))
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

if device == 'cuda':
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("\nTrain history saved to: ", "history.json")

print("\nF1 socre of the best model on the validation set is: ", best_f1) 
print("\nBest model saved to: ", "best_model.pth")