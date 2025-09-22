# Duygu Analizi ve Günlük Uygulaması

Bu proje, kullanıcıların günlük tutabileceği ve yazdıkları metinlerin duygu durum analizini yapabilen bir web uygulamasıdır. Kullanıcılar giriş yapıp günlüklerini yazabilir, yazdıkları metinlerin duygu durum analizini görebilir ve geçmiş girişlerini takip edebilirler.

## Özellikler

- **Kullanıcı Yönetimi**: Kayıt olma ve giriş yapma özellikleri
- **Duygu Analizi**: Yazılan metinlerin duygu durum analizi (Olumlu/Nötr/Olumsuz)
- **Günlük Takibi**: Tarih ve duygu durumuna göre günlük girişlerini filtreleme
- **Profil Yönetimi**: Kullanıcı bilgilerini görüntüleme ve güncelleme
- **Öneri Sistemi**: Kullanıcının ruh haline göre öneriler sunma

## Teknoloji Yığını

### Backend
- **Python 3.9+**
- **Flask**: Web uygulama çatısı
- **SQLAlchemy**: Veritabanı ORM'si
- **Transformers**: Duygu analizi için BERT modeli
- **SQLite**: Veritabanı

### Frontend
- **HTML5**
- **CSS3**
- **JavaScript**
- **Bootstrap 5**: Tasarım çerçevesi

## Kurulum

1. **Gereksinimlerin Yüklenmesi**
   ```bash
   pip install -r requirements.txt
   ```

2. **Veritabanı Kurulumu**
   ```bash
   cd backend
   python create_tables.py
   ```

3. **Uygulamayı Başlatma**
   ```bash
   python backend/app.py
   ```

4. **Tarayıcıda Açma**
   Uygulama `http://127.0.0.1:5000` adresinde başlatılacaktır.

## Proje Yapısı

```
EmotionalDetection/
├── backend/                  # Backend kaynak dosyaları
│   ├── __pycache__/
│   ├── app.py               # Ana Flask uygulaması
│   ├── config.py            # Yapılandırma ayarları
│   ├── create_tables.py     # Veritabanı tablolarını oluşturma
│   ├── models.py            # Veritabanı modelleri
│   └── predict_model.py     # Duygu analizi modeli
├── best_model/              # Eğitilmiş BERT modeli
├── frontend/                # Frontend dosyaları
│   ├── ProfilePage.html     # Kullanıcı profili sayfası
│   ├── journalPage.html     # Günlük sayfası
│   └── loginSignUpPage.html # Giriş/Kayıt sayfası
├── BertSentimentTrainer.py  # Model eğitim scripti
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Bu dosya
```

## Kullanım

1. **Kayıt Olma**
   - Ana sayfadan "Sign Up" butonuna tıklayın
   - Gerekli bilgileri doldurun ve hesap oluşturun

2. **Giriş Yapma**
   - E-posta ve şifrenizle giriş yapın

3. **Günlük Yazma**
   - "New Entry" butonuna tıklayın
   - Duygularınızı yazın ve "Save" butonuna basın
   - Sistem otomatik olarak duygu analizi yapacaktır

4. **Geçmiş Girişleri Görüntüleme**
   - Ana sayfada geçmiş girişleriniz listelenecektir
   - Tarih veya duygu durumuna göre filtreleme yapabilirsiniz

5. **Profil Yönetimi**
   - Sağ üst köşedeki profil resmine tıklayarak profilinizi görüntüleyebilir ve güncelleyebilirsiniz

## Geliştiriciler

- [İsminiz]
- [İletişim Bilgileriniz]

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## Katkıda Bulunma

1. Bu depoyu forklayın
2. Yeni özellik dalı oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi kaydedin (`git commit -m 'Add some AmazingFeature'`)
4. Dalınıza itin (`git push origin feature/AmazingFeature`)
5. Bir Pull Request açın
