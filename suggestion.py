from dotenv import load_dotenv
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests  # HTTP istekleri için
load_dotenv()

# --- Gemini API Key ---
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = os.getenv("GEMINI_API_URL")  # örnek endpoint, güncel URL’ye bak

class GeminiClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            # Önce gelen yanıtı yazdır
            print("Gemini yanıtı (debug):", result)

            # Güvenli parse
            candidates = result.get("candidates")
            if not candidates:
                return "Gemini’den cevap alınamadı."

            content = candidates[0].get("content")
            if not content:
                return "Gemini’den içerik alınamadı."

            # content artık dict, parts listesi var
            parts = content.get("parts", [])
            text_output = " ".join([part.get("text", "") for part in parts])
            return text_output

        else:
            raise Exception(f"Gemini API error: {response.status_code}, {response.text}")



# --- Recommender ---
class HobbyEmotionRecommender:
    def __init__(self, model, tokenizer, gemini_client):
        self.model = model
        self.tokenizer = tokenizer
        self.gemini_client = gemini_client

        if not hasattr(self.model.config, "id2label") or not self.model.config.id2label:
            self.model.config.id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear"}

    def predict_emotion(self, text):
        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                  max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_idx = logits.argmax(dim=-1).item()
        return self.model.config.id2label[predicted_idx]

    def generate_suggestions(self, user_text, predicted_emotion):
        prompt = f"""
Kullanıcının cevapları: {user_text}
Tahmin edilen duygu: {predicted_emotion}

Bu kullanıcının hobilerine ve duygu durumuna göre 5 öneri üret.
Öneriler kısa ve anlaşılır olsun. Madde madde 5 satır yaz sadece.
"""
        return self.gemini_client.chat(prompt)

# --- Kullanım ---
if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained("best_model").to("cpu")
    tokenizer = BertTokenizer.from_pretrained("best_model")

    gemini_client = GeminiClient(API_KEY)
    recommender = HobbyEmotionRecommender(model, tokenizer, gemini_client)

    questions = [
        "Hafta sonlarını nasıl değerlendiriyorsun?",
        "Boş zamanlarında ne yapmaktan hoşlanırsın?",
        "Yeni bir şeyler öğrenmeye açık mısın?",
        "Arkadaşlarınla vakit geçirirken neler yaparsın?",
        "Spor yapıyor musun? Hangi türleri?",
        "Sanat veya müzikle ilgileniyor musun?",
        "Kendi başına vakit geçirirken neler yaparsın?",
        "Seyahat etmeyi sever misin?",
        "Hangi aktiviteler seni motive eder?",
        "Hobilerini anlatır mısın?"
    ]

    user_answers = [input(q + " ") for q in questions]
    full_text = " ".join(user_answers)

    predicted_emotion = recommender.predict_emotion(full_text)
    print(f"\nTahmin edilen duygu: {predicted_emotion}")

    suggestions = recommender.generate_suggestions(full_text, predicted_emotion)
    print("\nÖneriler:")
    print(suggestions)