=== Binary Değerlendirme ===
              precision    recall  f1-score   support

      Benign       0.90      0.98      0.94    232650
   Malicious       0.93      0.73      0.82     94242

    accuracy                           0.91    326892
   macro avg       0.91      0.85      0.88    326892
weighted avg       0.91      0.91      0.90    326892

Accuracy: 0.9064
Balanced Accuracy: 0.8547
Jaccard Score: 0.6929
F1 Score: 0.8186
ROC-AUC Score: 0.8547
Gini: 0.7094
Matthews CorrCoef (MCC): 0.7663

=== Multiclass Değerlendirme ===
              precision    recall  f1-score   support

  defacement       0.95      0.95      0.95     28836
     malware       0.89      0.86      0.88     11056
    phishing       0.92      0.92      0.92     20343

    accuracy                           0.93     60235
   macro avg       0.92      0.91      0.91     60235
weighted avg       0.93      0.93      0.93     60235

Cohen's Kappa: 0.8807
Hamming Loss: 0.0741
Accuracy: 0.9259
Balanced Accuracy: 0.9123
Jaccard Score (macro): 0.8445
Matthews CorrCoef (MCC): 0.8808
Log Loss: 2.6718

=== End-to-End Sistem Değerlendirmesi ===
              precision    recall  f1-score   support

      benign       0.96      0.98      0.97    232650
  defacement       0.92      0.94      0.93     29289
     malware       0.85      0.85      0.85     11254
    phishing       0.76      0.64      0.69     29463

    accuracy                           0.94    302656
   macro avg       0.87      0.85      0.86    302656
weighted avg       0.93      0.94      0.93    302656

Cohen's Kappa: 0.8291
Hamming Loss: 0.0648
Accuracy: 0.9352
Balanced Accuracy: 0.8498
Jaccard Score (macro): 0.7689
Matthews CorrCoef (MCC): 0.8299
Log Loss: 2.3354