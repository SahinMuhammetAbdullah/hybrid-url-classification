import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier  # type: ignore

# ----- Dosya Yollarını Ayarla -----
data_path = "data/input_data.csv"

# ----- 1. Adım: Veri Seti Hazırlığı -----
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"

df['binary_label'] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)

# ----- 2. Adım: XGBoost ile Zararlı mı? Tespiti -----
X = df.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
y_binary = df['binary_label']

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

xgb_binary = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric="logloss")
xgb_binary.fit(X_train, y_train)

y_pred_xgb_binary = xgb_binary.predict(X_test)
print("\nXGBoost Modeli Performansı (Zararlı mı?):")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb_binary, target_names=['Zararsız', 'Zararlı']))

import joblib

# Modeli kaydet
joblib.dump(xgb_binary, 'binary-pkl/xgb_binary_model.pkl')

# Kaydedilen modeli yüklemek için
# loaded_model = joblib.load('xgb_binary_model.pkl')
# y_pred = loaded_model.predict(X_test)
