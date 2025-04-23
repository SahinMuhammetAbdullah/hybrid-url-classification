=== Binary Değerlendirme ===
              precision    recall  f1-score   support

      Benign       0.97      0.99      0.98    232650
   Malicious       0.97      0.92      0.94     94242

    accuracy                           0.97    326892
   macro avg       0.97      0.95      0.96    326892
weighted avg       0.97      0.97      0.97    326892

Accuracy: 0.9677
Balanced Accuracy: 0.9525
Jaccard Score: 0.8912
F1 Score: 0.9425
ROC-AUC Score: 0.9525
Gini: 0.9050
Matthews CorrCoef (MCC): 0.9208

=== Multiclass Değerlendirme ===
              precision    recall  f1-score   support

  defacement       0.94      0.95      0.95     29266
     malware       0.88      0.86      0.87     11219
    phishing       0.93      0.93      0.93     26465

    accuracy                           0.93     66950
   macro avg       0.92      0.92      0.92     66950
weighted avg       0.93      0.93      0.93     66950

Cohen's Kappa: 0.8865
Hamming Loss: 0.0708
Accuracy: 0.9292
Balanced Accuracy: 0.9156
Jaccard Score (macro): 0.8489
Matthews CorrCoef (MCC): 0.8865
Log Loss: 2.5524

=== End-to-End Sistem Değerlendirmesi ===
              precision    recall  f1-score   support

      benign       0.99      0.99      0.99    232650
  defacement       0.94      0.95      0.95     29289
     malware       0.87      0.86      0.87     11254
    phishing       0.85      0.84      0.85     29463

    accuracy                           0.97    302656
   macro avg       0.91      0.91      0.91    302656
weighted avg       0.97      0.97      0.97    302656

Cohen's Kappa: 0.9108
Hamming Loss: 0.0346
Accuracy: 0.9654
Balanced Accuracy: 0.9092
Jaccard Score (macro): 0.8421
Matthews CorrCoef (MCC): 0.9108
Log Loss: 1.2480