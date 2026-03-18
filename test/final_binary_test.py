import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ----- Argümanları Tanımla -----
parser = argparse.ArgumentParser(description="İkili Model Test Aracı")
parser.add_argument("-binary", type=str, required=True, help="Model tipi: rf, svm, xgb, vb.")
args = parser.parse_args()

# ----- Dosya Yolları -----
data_path = "data/cleaned_feature_data.csv"
model_dir = "binary-model-save"

# ----- Veri Seti Yükleme ve Hazırlama -----
print(f"Veri yükleniyor: {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"HATA: Veri dosyası bulunamadı: {data_path}")
    exit()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"

# Etiketleri Türkçeleştirme ve İkili Etiket Oluşturma
df['binary_label'] = df[target_column].apply(lambda x: 0 if x.lower() in ["benign", "good", "güvenli"] else 1)

# Sayısal özellikleri ve hedef değişkeni ayır
X = df.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
y = df['binary_label']

# ----- %30 TEST AYRIMI -----
# random_state=42: Her çalıştırdığında aynı test verilerinin gelmesini sağlar.
# stratify=y: Güvenli/Zararlı oranını korur.
_, X_test, _, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Toplam Veri: {len(X)} | Test Verisi (%30): {len(X_test)}")

# ----- Model Yükleme -----
model_name = f"{model_dir}/{args.binary}_binary_model.pkl"
zip_model_name = f"{model_dir}/{args.binary}_binary_model.zip"

try:
    model = joblib.load(model_name)
    print(f"Model başarıyla yüklendi: {model_name}")
except FileNotFoundError:
    try:
        model = joblib.load(zip_model_name)
        print(f"Model başarıyla yüklendi: {zip_model_name}")
    except FileNotFoundError:
        print(f"HATA: {args.binary} modeli bulunamadı.")
        exit()

# ----- Tahmin ve Raporlama -----
print(f"Tahminler yapılıyor... (Test Seti Üzerinde)")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred, average='weighted') * 100
rec = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print("\n" + "="*60)
print(f"İkili Model Test Sonuçları (Yüzde 30 Test): {args.binary.upper()}")
print("="*60)
print(f"Doğruluk (Accuracy)    : {acc:.6f}%")
print(f"Keskinlik (Precision)  : {prec:.6f}%")
print(f"Duyarlılık (Recall)    : {rec:.6f}%")
print(f"F1-Skoru (F1-Score)    : {f1:.6f}%")
print("="*60)

print("\n--- Detaylı Sınıflandırma Raporu (Test Verisi) ---")
print(classification_report(y_test, y_pred, target_names=["Güvenli (0)", "Zararlı (1)"]))

# ----- KARMAŞIKLIK MATRİSİ -----
cm = confusion_matrix(y_test, y_pred)
labels = ["Güvenli (0)", "Zararlı (1)"]



plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,       
    fmt='d',          
    cmap='Blues',     
    xticklabels=labels,
    yticklabels=labels,
    linewidths=.5,
    linecolor='lightgrey'
)

plt.title(f'{args.binary.upper()} İkili Model Karmaşıklık Matrisi', fontsize=14, pad=20)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.tight_layout()
plt.show()

print("\nİşlem tamamlandı. Test seti sonuçları yukarıdadır.")