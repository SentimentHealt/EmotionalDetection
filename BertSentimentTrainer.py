import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler, pipeline
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
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
    def __init__(self, model_name="dbmdz/bert-base-turkish-uncased", lr=2e-5, epochs=3, batch_size=16, patience=3, weight_decay=0, warmup=0):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.warmup = warmup

    def prepare_data(self, train_path, val_path=None, test_path=None, text_col="text", label_col="label"):
        # Eğitim verisi
        df_train = pd.read_csv(train_path)
        df_train.columns = df_train.columns.str.strip()  # boşlukları temizle

        # Validation
        if val_path:
            df_val = pd.read_csv(val_path)
            df_val.columns = df_val.columns.str.strip()
        else:
            df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

        # Test
        if test_path:
            df_test = pd.read_csv(test_path)
            df_test.columns = df_test.columns.str.strip()
        else:
            df_test = None

        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_train["label_encoded"] = le.fit_transform(df_train[label_col])
        df_val["label_encoded"] = le.transform(df_val[label_col])
        if df_test is not None:
            df_test["label_encoded"] = le.transform(df_test[label_col])

        self.label2id = dict(zip(le.classes_, le.transform(le.classes_)))
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id)
        ).to(self.device)

        # Dataset ve DataLoader
        train_ds = EmotionDataset(df_train[text_col].tolist(), df_train["label_encoded"].tolist(), self.tokenizer)
        val_ds = EmotionDataset(df_val[text_col].tolist(), df_val["label_encoded"].tolist(), self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        test_loader = None
        if df_test is not None:
            test_ds = EmotionDataset(df_test[text_col].tolist(), df_test["label_encoded"].tolist(), self.tokenizer)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        num_training_steps = self.epochs * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(self.warmup*num_training_steps), num_training_steps=num_training_steps)

        best_val_loss = float("inf")
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

            print(f"Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_correct/len(train_loader.dataset):.4f}")

            val_loss, _ = self.evaluate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping tetiklendi.")
                    break

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                total_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / len(val_loader.dataset)
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def save_model(self, path="saved_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model ve tokenizer '{path}' klasörüne kaydedildi.")


if __name__ == "__main__":
    train_path = "/training.csv"
    val_path = "/validation.csv"
    test_path = "/test.csv"

    clf = BertSentimentTrainer(epochs=20, lr=3e-5, patience=2)
    train_loader, val_loader, test_loader = clf.prepare_data(train_path, val_path, test_path, text_col="text", label_col="label")
    clf.train(train_loader, val_loader)

    if test_loader:
        print("\nTest set üzerinde değerlendirme:")
        clf.evaluate(test_loader)
