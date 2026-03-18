import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Test Başlatılıyor: DQN Çok Sınıflı Analiz (%30 Test - Hızlandırılmış) ---")

# === 1. Veri Yükleme ve Hazırlama ===
DATA_PATH = "data/cleaned_feature_data.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"HATA: Veri dosyası bulunamadı: {DATA_PATH}")
    exit()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"

# Etiketleri Belirttiğin Şekilde Türkçeleştirme
tr_map = {
    "good": "Güvenli",
    "benign": "Güvenli",
    "phishing": "Oltalama",
    "malware": "Zararlı Yazılım",
    "defacement": "Tahrif",
    "spam": "Spam"
}

df[target_column] = df[target_column].str.lower().replace(tr_map)

# Sayısal özellikleri ve hedef değişkeni ayır
X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
y = df[target_column]

# === %30 TEST AYRIMI ===
# Önce tüm veri setini bölüyoruz ki 'Güvenli' ve 'Zararlı' dengesi korunsun
_, X_test_raw, _, y_test_raw = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Test seti içinden sadece ZARARLI olanları filtrele (Katman 2 testi için)
malicious_mask = y_test_raw != "Güvenli"
X_test = X_test_raw[malicious_mask].copy()
y_test = y_test_raw[malicious_mask].values

print(f"Toplam Test Verisi: {len(X_test_raw)} | Test İçindeki Zararlı Sayısı: {len(X_test)}")

if len(X_test) == 0:
    print("HATA: Test seti içinde zararlı veri bulunamadı!")
    exit()

# === 2. Model ve Etiket Haritasını Yükle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = DQN.load("multiclass-model-save/multiclass_dqn_model", device=device)
    with open("multiclass-model-save/multiclass_labels.json", "r") as f:
        label_map_raw = json.load(f)
except FileNotFoundError as e:
    print(f"HATA: Gerekli dosyalar bulunamadı: {e}")
    exit()

# === 3. Hızlandırılmış Tahmin (Batch Prediction) ===
print(f"Tahminler yapılıyor... (Cihaz: {device})")

# Veriyi tek seferde Tensor'a çevirip modelin Q-Network'üne gönderiyoruz
obs_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

with torch.no_grad():
    q_values = model.q_net(obs_tensor)
    actions = torch.argmax(q_values, dim=1).cpu().numpy()

# Tahmin edilen aksiyon numaralarını Türkçe etiketlere çevir
y_pred = [tr_map.get(label_map_raw.get(str(a)).lower(), "Bilinmeyen") for a in actions]

# === 4. Raporlama (6 Basamaklı Hassasiyet) ===
valid_labels = ["Oltalama", "Zararlı Yazılım", "Tahrif", "Spam"]

print("\n" + "="*60)
print("ÇOK SINIFLI SINIFLANDIRMA RAPORU")
print("="*60)
print(classification_report(y_test, y_pred, labels=valid_labels, zero_division=0))

report_dict = classification_report(y_test, y_pred, labels=valid_labels, zero_division=0, output_dict=True)

# İstatistikleri Yazdır
accuracy = accuracy_score(y_test, y_pred) * 100
precision = report_dict["weighted avg"]["precision"] * 100
recall = report_dict["weighted avg"]["recall"] * 100
f1 = report_dict["weighted avg"]["f1-score"] * 100

print(f"Doğruluk (Accuracy)    : {accuracy:.6f}%")
print(f"Keskinlik (Precision)  : {precision:.6f}%")
print(f"Duyarlılık (Recall)    : {recall:.6f}%")
print(f"F1-Skoru (F1-Score)    : {f1:.6f}%")
print("="*60)

# === 5. Karmaşıklık Matrisi (Açık Renk - Blues) ===
cm = confusion_matrix(y_test, y_pred, labels=valid_labels)



plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,         
    fmt='d',            
    cmap='Blues',       
    xticklabels=valid_labels, 
    yticklabels=valid_labels,
    linewidths=.5,
    linecolor='lightgrey'
)
plt.title('DQN Çok Sınıflı Karmaşıklık Matrisi', fontsize=14, pad=20)
plt.xlabel('Tahmin Edilen Tür', fontsize=12)
plt.ylabel('Gerçek Tür', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nİşlem başarıyla tamamlandı.")