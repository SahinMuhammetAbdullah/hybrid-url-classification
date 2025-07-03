import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import joblib

# ----- Dosya Yollarını Ayarla -----
data_path = "data/cleaned_dataV4.csv"

# ----- 1. Adım: Veri Seti Hazırlığı -----
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
df['binary_label'] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)

X = df.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
y_binary = df['binary_label']

# ----- 2. Adım: SVM ile Zararlı mı? Tespiti -----
df_binary = df[df['binary_label'].isin([0, 1])]
y_binary = df_binary['binary_label']
X_binary = df_binary.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

# SVC modelini oluştur (RBF kernel varsayılan)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Tahmin yap
y_pred_svm = svm_model.predict(X_test)

# Performans raporu
print("\nSVM Modeli Performansı (Zararlı mı?):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=['Zararsız', 'Zararlı']))

# Modeli kaydet
joblib.dump(svm_model, 'binary-pkl/svm_binary_model.pkl')

# Kaydettiğin modeli daha sonra şöyle yükleyebilirsin:
# loaded_svm_model = joblib.load('binary-pkl/svm_binary_model.pkl')
# y_pred = loaded_svm_model.predict(X_test)
