import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler, pipeline
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
    def __init__(self, model_name="bert-base-uncased", lr=2e-5, epochs=3, batch_size=16, patience=3, test_size=0.2, weight_decay=0, warmup=0):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None #Model daha sonra kurulacak
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience #Early stopping
        self.test_size = test_size
        self.weight_decay = weight_decay
        self.warmup = warmup

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
            df["content"].tolist(), df["label"].tolist(), test_size=self.test_size, random_state=42
        )

        train_ds = EmotionDataset(train_texts, train_labels, self.tokenizer)
        val_ds = EmotionDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        num_training_steps = self.epochs * len(train_loader)
        num_warmup_steps = int(self.warmup * num_training_steps)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0

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

            # Validation
            val_loss, _ = self.evaluate(val_loader)

            # Early Stopping kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                # En iyi modeli kaydet
                self.save_model("best_model")
                print(f"Validation loss iyileşti (Epoch {best_epoch}), model kaydedildi.")
            else:
                patience_counter += 1
                print(f"Validation loss iyileşmedi ({patience_counter}/{self.patience})")

                if patience_counter >= self.patience:
                    print("Early stopping tetiklendi.")
                    break

        print(f"\nEğitim bitti! En iyi model Epoch {best_epoch} sırasında bulundu.")

        # En iyi modeli yükleyip örnek tahmin
        best_model = BertForSequenceClassification.from_pretrained("best_model").to(self.device)
        clf = pipeline("text-classification", model=best_model, tokenizer=self.tokenizer, device=-1)
        print("Örnek tahmin:", clf("I haven't been able to think about much for a few days."))

    def evaluate(self, val_loader):
        self.model.eval()
        total_correct, total_loss = 0, 0

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
        return avg_loss, accuracy

    def save_model(self, path="saved_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model ve tokenizer '{path}' klasörüne kaydedildi.")



if __name__ == "__main__":
    csv_path = "/Users/iclalhoruz/Downloads/tweet_emotions.csv"
    clf = BertSentimentTrainer(epochs=10, lr=3e-5, patience=2, test_size=0.3, weight_decay=0.01, warmup=0.06)
    train_loader, val_loader = clf.prepare_data(csv_path)
    clf.train(train_loader, val_loader)

