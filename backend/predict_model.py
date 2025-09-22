# predict_model.py
import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH=os.getenv("MODEL_PATH")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Senin label mapping'in
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
}

def predict_emotion(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(dim=-1).item()
    return id2label[pred_id]
