import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import joblib
import time # Zaman ölçümü için eklendi
import os # Dizin oluşturmak için eklendi

# ----- Dosya Yollarını Ayarla -----
data_path = "data/cleaned_feature_data.csv"
model_save_path = 'binary-pkl/'

# Model kaydetme dizinini oluştur (eğer yoksa)
os.makedirs(model_save_path, exist_ok=True)
print(f"Model kayıt dizini '{model_save_path}' kontrol edildi/oluşturuldu.")

print("----- 1. Adım: Veri Seti Hazırlığı -----")

print(f"Veri yükleniyor: {data_path}...")
try:
    df = pd.read_csv(data_path)
    print(f"Veri başarıyla yüklendi. Boyut: {df.shape}")
except FileNotFoundError:
    print(f"HATA: Veri dosyası bulunamadı: {data_path}")
    exit()
except Exception as e:
    print(f"HATA: Veri yüklenirken bir sorun oluştu: {e}")
    exit()

print("NaN ve Inf değerleri temizleniyor...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Temizlenmiş veri boyutu: {df.shape}")

if df.empty:
    print("HATA: Veri seti temizleme sonrası boş kaldı. Lütfen verinizi kontrol edin.")
    exit()

target_column = "URL_Type_obf_Type"
print(f"Hedef sütun: '{target_column}'")

print("Binary etiketler oluşturuluyor ('binary_label')...")
# 'benign' veya 'good' olanları 0, diğerlerini 1 olarak etiketle
df['binary_label'] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)
print("Binary etiket dağılımı:")
print(df['binary_label'].value_counts(normalize=True))

print("Özellikler (X) ve hedef (y_binary) ayrılıyor...")
# Sadece sayısal sütunları al
X = df.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
y_binary = df['binary_label']

if X.empty:
    print("HATA: Özellik matrisi (X) boş. Sayısal sütun bulunamadı veya hepsi çıkarıldı.")
    exit()

print(f"Özellik matrisi (X) boyutu: {X.shape}")
print(f"Hedef vektör (y_binary) boyutu: {y_binary.shape}")


# ----- 2. Adım: SVM ile Zararlı mı? Tespiti -----
# df_binary ve sonraki X_binary, y_binary tanımlamaları gereksiz görünüyor,
# çünkü df zaten 'binary_label' içeriyor ve NaN'lar temizlendi.
# Direkt yukarıda oluşturulan X ve y_binary kullanılabilir.
# Eğer df['binary_label'] içinde 0 ve 1 dışında değerler olsaydı bu filtreleme (isin([0,1])) anlamlı olurdu.
# Ancak apply fonksiyonu zaten sadece 0 ve 1 üretiyor.
# Bu yüzden aşağıdaki satırları yorumluyorum ve yukarıdaki X, y_binary'yi kullanıyorum:
# df_binary = df[df['binary_label'].isin([0, 1])] # Bu satır, eğer apply sonucu NaN vs olsaydı anlamlı olurdu
# y_binary_filtered = df_binary['binary_label']
# X_binary_filtered = df_binary.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
# X_train, X_test, y_train, y_test = train_test_split(X_binary_filtered, y_binary_filtered, test_size=0.3, random_state=42)

print("Veri eğitim ve test setlerine ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary) # stratify eklendi
print(f"Eğitim seti boyutu: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test seti boyutu: X_test={X_test.shape}, y_test={y_test.shape}")
print("Eğitim seti etiket dağılımı:\n", y_train.value_counts(normalize=True))
print("Test seti etiket dağılımı:\n", y_test.value_counts(normalize=True))


print("\n----- SVM Modeli Eğitimi -----")
# SVC modelini oluştur (RBF kernel varsayılan)
# verbose=True ekleyerek eğitim sırasında ilerleme bilgisi alabilirsiniz
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=True) # verbose=True eklendi, shrinking=False (Warning: using -h 0 may be faster için kullanılabilir)

print("SVM modeli eğitiliyor (bu işlem uzun sürebilir)...")
start_time = time.time()
try:
    svm_model.fit(X_train, y_train)
except Exception as e:
    print(f"HATA: Model eğitimi sırasında bir sorun oluştu: {e}")
    exit()
end_time = time.time()
training_time = end_time - start_time
print(f"SVM modeli eğitimi tamamlandı. Süre: {training_time:.2f} saniye.")


print("\n----- Model Performansı Değerlendirme -----")
print("Test seti üzerinde tahmin yapılıyor...")
start_time_pred = time.time()
y_pred_svm = svm_model.predict(X_test)
end_time_pred = time.time()
prediction_time = end_time_pred - start_time_pred
print(f"Tahmin süresi: {prediction_time:.4f} saniye.")

# Performans raporu
print("\nSVM Modeli Performansı (Zararlı mı?):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=['Zararsız', 'Zararlı']))


print("\n----- Model Kaydetme -----")
model_filename = os.path.join(model_save_path, 'svm_binary_model.pkl')
try:
    joblib.dump(svm_model, model_filename)
    print(f"Model başarıyla '{model_filename}' olarak kaydedildi.")
except Exception as e:
    print(f"HATA: Model kaydedilirken bir sorun oluştu: {e}")

print("\nİşlem tamamlandı.")

# Kaydettiğin modeli daha sonra şöyle yükleyebilirsin:
# print("\n----- Model Yükleme (Örnek) -----")
# try:
#     loaded_svm_model = joblib.load(model_filename)
#     print(f"Model '{model_filename}' başarıyla yüklendi.")
#     # Örnek bir tahmin
#     # y_pred_loaded = loaded_svm_model.predict(X_test)
#     # print("Yüklenen model ile örnek tahmin yapıldı.")
# except FileNotFoundError:
#     print(f"HATA: Kayıtlı model dosyası bulunamadı: {model_filename}")
# except Exception as e:
#     print(f"HATA: Model yüklenirken bir sorun oluştu: {e}")