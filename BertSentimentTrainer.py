import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class BertSentimentTrainer:
    def __init__(self, model_name="bert-base-uncased", lr=2e-5, epochs=3, batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None #Model daha sonra kurulacak
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["sentiment"])
        self.label2id = dict(zip(le.classes_, le.transform(le.classes_)))
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id)
        ).to(self.device)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["content"].tolist(), df["label"].tolist(), test_size=0.1, random_state=42
        )

        train_ds = EmotionDataset(train_texts, train_labels, self.tokenizer)
        val_ds = EmotionDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_training_steps = self.epochs * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            self.model.train()
            total_loss, total_correct = 0, 0

            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = total_correct / (len(train_loader.dataset))
            print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.evaluate(val_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_correct, total_loss = 0, 0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / len(val_loader.dataset)
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    csv_path = "/Users/iclalhoruz/Downloads/tweet_emotions.csv"
    clf = BertSentimentTrainer(epochs=2)
    train_loader, val_loader = clf.prepare_data(csv_path)
    clf.train(train_loader, val_loader)

