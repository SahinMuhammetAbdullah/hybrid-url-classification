=== Binary Değerlendirme ===
              precision    recall  f1-score   support

      Benign       0.92      0.95      0.94    232650
   Malicious       0.87      0.80      0.83     94242

    accuracy                           0.91    326892
   macro avg       0.89      0.88      0.88    326892
weighted avg       0.91      0.91      0.91    326892

Accuracy: 0.9073
Balanced Accuracy: 0.8753
Jaccard Score: 0.7132
F1 Score: 0.8326
ROC-AUC Score: 0.8753
Gini: 0.7505
Matthews CorrCoef (MCC): 0.7699

=== Multiclass Değerlendirme ===
              precision    recall  f1-score   support

  defacement       0.94      0.95      0.95     28090
     malware       0.88      0.87      0.87     10624
    phishing       0.93      0.93      0.93     23037

    accuracy                           0.93     61751
   macro avg       0.92      0.92      0.92     61751
weighted avg       0.93      0.93      0.93     61751

Cohen's Kappa: 0.8848
Hamming Loss: 0.0718
Accuracy: 0.9282
Balanced Accuracy: 0.9150
Jaccard Score (macro): 0.8479
Matthews CorrCoef (MCC): 0.8848
Log Loss: 2.5881

=== End-to-End Sistem Değerlendirmesi ===
              precision    recall  f1-score   support

      benign       0.96      0.95      0.96    232650
  defacement       0.93      0.91      0.92     29289
     malware       0.85      0.82      0.83     11254
    phishing       0.64      0.72      0.68     29463

    accuracy                           0.92    302656
   macro avg       0.84      0.85      0.85    302656
weighted avg       0.92      0.92      0.92    302656

Cohen's Kappa: 0.7987
Hamming Loss: 0.0797
Accuracy: 0.9203
Balanced Accuracy: 0.8516
Jaccard Score (macro): 0.7495
Matthews CorrCoef (MCC): 0.7991
Log Loss: 2.8721