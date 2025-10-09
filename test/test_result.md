--- End-to-End Performance (Binary: RF) ---
              precision    recall  f1-score   support

      benign       1.00      1.00      1.00    141012
  defacement       0.92      0.95      0.94     19843
     malware       0.97      0.79      0.87     11114
    phishing       0.92      0.96      0.94     51860
        spam       0.95      0.91      0.93     14643

    accuracy                           0.97    238472
   macro avg       0.95      0.92      0.93    238472
weighted avg       0.97      0.97      0.97    238472


--- Weighted Average (Percentage Format) ---
Precision : 96.906916%
Recall    : 96.857493%
F1-Score  : 96.828475%
Accuracy  : 96.857493%

--- Eğitim Tamamlandı: Son Test ---
              precision    recall  f1-score   support

  defacement       0.92      0.95      0.94     65606
     malware       0.97      0.79      0.87     36979
    phishing       0.92      0.96      0.94    173509
        spam       0.95      0.91      0.93     48774

    accuracy                           0.93    324868
   macro avg       0.94      0.90      0.92    324868
weighted avg       0.93      0.93      0.93    324868


--- Weighted Average Results ---
Precision : 93.347360%
Recall    : 93.227403%
F1-Score  : 93.144603%
Accuracy  : 93.227403%

=== Binary Model Test Sonuçları ===
Model: RF
Doğruluk (Accuracy): 99.556803
Kesinlik (Precision): 99.556888
Duyarlılık (Recall): 99.556803
F1-Skoru: 99.556832

--- Ayrıntılı Classification Report ---
              precision    recall  f1-score   support

    Zararsız       1.00      1.00      1.00    470038
     Zararlı       0.99      1.00      0.99    324868

    accuracy                           1.00    794906
   macro avg       1.00      1.00      1.00    794906
weighted avg       1.00      1.00      1.00    794906

--- Filtered End-to-End Performance (RF True Positives to DQN) ---
              precision    recall  f1-score   support

      benign       1.00      1.00      1.00    141012
  defacement       0.92      0.95      0.94     19843
     malware       0.97      0.79      0.87     11114
    phishing       0.92      0.96      0.94     51860
        spam       0.96      0.91      0.93     14643

    accuracy                           0.97    238472
   macro avg       0.95      0.92      0.94    238472
weighted avg       0.97      0.97      0.97    238472


--- Weighted Average (Percentage Format) ---
Accuracy  : 97.0865%
Precision : 97.1199%
Recall    : 97.0865%
F1-Score  : 97.0497%

Çalıştırılıyor: Filtreli Hibrit Boru Hattı (Binary: RF, Yalnızca Gerçek Zararlı Doğru Pozitifler DQN'e)...

--- Filtreli Uçtan Uca Performans (Binary: RF) ---
              precision    recall  f1-score   support

      benign       1.00      1.00      1.00    141012
  defacement       0.92      0.95      0.94     19843
     malware       0.97      0.79      0.87     11114
    phishing       0.92      0.96      0.94     51860
        spam       0.96      0.91      0.93     14643

    accuracy                           0.97    238472
   macro avg       0.95      0.92      0.94    238472
weighted avg       0.97      0.97      0.97    238472


--- Weighted Average (Percentage Format) ---
Accuracy  : 97.0865%
Precision : 97.1199%
Recall    : 97.0865%
F1-Score  : 97.0497%