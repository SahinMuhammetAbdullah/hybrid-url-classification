## 🧠 Binary Testler

| Test Adı               | Kullanım Amacı                                                        | Değer Aralığı    | Anlamı & Aksiyon                                                                                   |
|------------------------|------------------------------------------------------------------------|------------------|----------------------------------------------------------------------------------------------------|
| Accuracy               | Doğru tahmin edilen örneklerin tüm örneklere oranı                    | 0.0 - 1.0        | > 0.8: Yüksek doğruluk ✅ <br> 0.6 - 0.8: Kabul edilebilir ⚠️ <br> < 0.6: Model başarısız ❌         |
| Balanced Accuracy      | Her sınıf için hesaplanan doğruluk ortalaması                         | 0.0 - 1.0        | > 0.8: Dengeli ve doğru model ✅ <br> 0.6 - 0.8: Dengesiz veri ihtimali ⚠️ <br> < 0.6: Zayıf model ❌|
| F1 Score               | Precision ve Recall'ın harmonik ortalaması                            | 0.0 - 1.0        | > 0.8: Dengeli model ✅ <br> 0.6 - 0.8: Geliştirilebilir ⚠️ <br> < 0.6: Dengesizlik sorunu ❌        |
| ROC-AUC Score          | Sınıflar arası ayrımı ölçer (Receiver Operating Characteristic)       | 0.5 - 1.0        | > 0.8: Çok başarılı ayrım ✅ <br> 0.6 - 0.8: Sınırlı ayrım ⚠️ <br> < 0.6: Ayrım yetersiz ❌          |
| Gini Coefficient       | Ayrım kalitesini ROC-AUC üzerinden ölçer                              | 0.0 - 1.0        | > 0.6: Güçlü ayrım ✅ <br> 0.4 - 0.6: İyileştirilmeli ⚠️ <br> < 0.4: Veri/model hatası ❌            |
| Matthews CorrCoef (MCC)| Dengesiz veri setlerinde doğruluğu daha iyi yansıtır                  | -1.0 - 1.0       | > 0.6: Sağlam model ✅ <br> 0.4 - 0.6: Kararsız ⚠️ <br> < 0.4: Başarısız tahmin ❌                    |
| Jaccard Score          | Gerçek ve tahmin kesişiminin birleşime oranı                          | 0.0 - 1.0        | > 0.7: Yüksek benzerlik ✅ <br> 0.5 - 0.7: Kabul edilebilir ⚠️ <br> < 0.5: Benzerlik düşük ❌        |
| Confusion Matrix       | Sınıf tahminlerinin doğru ve yanlışlarını görselleştirir              | -                | Karışıklık noktaları netleşir. 🔍 Hangi sınıf daha çok karışıyor gözlemlenir.                      |

---

## 🏷️ Multiclass Testler

| Test Adı               | Kullanım Amacı                                                        | Değer Aralığı    | Anlamı & Aksiyon                                                                                   |
|------------------------|------------------------------------------------------------------------|------------------|----------------------------------------------------------------------------------------------------|
| Accuracy               | Tüm sınıflar için genel doğruluk oranı                                | 0.0 - 1.0        | > 0.8: Güçlü tahmin ✅ <br> 0.6 - 0.8: Geliştirilebilir ⚠️ <br> < 0.6: Yetersiz başarı ❌             |
| Balanced Accuracy      | Her sınıf için ayrı ayrı hesaplanan doğruluk ortalaması                | 0.0 - 1.0        | > 0.8: Dengeli başarı ✅ <br> 0.6 - 0.8: Bazı sınıflar eksik ⚠️ <br> < 0.6: Ciddi dengesizlik ❌       |
| F1 Score (macro)       | Her sınıfın eşit ağırlıkla F1 ortalaması                              | 0.0 - 1.0        | > 0.8: Dengeli performans ✅ <br> 0.6 - 0.8: Kısmen başarılı ⚠️ <br> < 0.6: İyileştirme gerekli ❌    |
| Cohen’s Kappa          | Tahminlerin rastgele olma ihtimaline karşı uyum ölçer                 | -1.0 - 1.0       | > 0.6: Güçlü uyum ✅ <br> 0.4 - 0.6: Kararsız uyum ⚠️ <br> < 0.4: Rastgeleye yakın ❌                 |
| Hamming Loss           | Hatalı tahmin edilen sınıfların oranı                                | 0.0 - 1.0        | < 0.2: Düşük hata ✅ <br> 0.2 - 0.4: Orta seviye ⚠️ <br> > 0.4: Yüksek hata ❌                        |
| Jaccard Score (macro)  | Her sınıf için ortalama benzerlik oranı                              | 0.0 - 1.0        | > 0.7: Yüksek benzerlik ✅ <br> 0.5 - 0.7: Orta düzey ⚠️ <br> < 0.5: Düşük başarı ❌                  |
| Matthews CorrCoef (MCC)| Sınıflandırmanın genel tutarlılığını ölçer                            | -1.0 - 1.0       | > 0.6: Güçlü model ✅ <br> 0.4 - 0.6: Geliştirilmeli ⚠️ <br> < 0.4: Tutarsız model ❌                 |
| Log Loss               | Tahminlerin ne kadar belirsiz olduğunu ölçer                         | 0.0 - ∞          | < 0.5: Net tahmin ✅ <br> 0.5 - 1.0: Belirsiz ⚠️ <br> > 1.0: Kararsız model ❌                       |
| Classification Report  | Her sınıf için Precision, Recall ve F1 skorlarını detaylı verir       | 0.0 - 1.0        | > 0.8: Dengeli dağılım ✅ <br> 0.6 - 0.8: Detaylı analiz yapılmalı ⚠️ <br> < 0.6: Başarı zayıf ❌    |

---

## 🔁 End-to-End Sistem Testi

| Test Adı               | Kullanım Amacı                                                        | Değer Aralığı    | Anlamı & Aksiyon                                                                                   |
|------------------------|------------------------------------------------------------------------|------------------|----------------------------------------------------------------------------------------------------|
| Accuracy               | Uçtan uca sistemin genel doğruluğu                                    | 0.0 - 1.0        | > 0.8: Sağlam sistem ✅ <br> 0.6 - 0.8: Geliştirilebilir ⚠️ <br> < 0.6: Zayıf performans ❌          |
| Balanced Accuracy      | Her sınıfın başarı ortalaması                                         | 0.0 - 1.0        | > 0.8: Tüm sistem dengeli ✅ <br> 0.6 - 0.8: Kısmen dengesiz ⚠️ <br> < 0.6: Gözden geçirilmeli ❌     |
| Jaccard Score (macro)  | Genel sınıflar arası benzerlik oranı                                 | 0.0 - 1.0        | > 0.7: Benzerlik yüksek ✅ <br> 0.5 - 0.7: Kabul edilebilir ⚠️ <br> < 0.5: Belirsiz sonuçlar ❌      |
| Cohen’s Kappa          | Gerçek etiketlerle tahminlerin genel uyumu                           | -1.0 - 1.0       | > 0.6: Güçlü sistem uyumu ✅ <br> 0.4 - 0.6: Dengesiz ⚠️ <br> < 0.4: Rastgeleye yakın ❌             |
| Hamming Loss           | Uçtan uca sistemin hata oranı                                         | 0.0 - 1.0        | < 0.2: Başarılı sistem ✅ <br> 0.2 - 0.4: İyileştirilebilir ⚠️ <br> > 0.4: Düşük güven ❌            |
| Matthews CorrCoef (MCC)| Sistemin genel tahmin tutarlılığı                                    | -1.0 - 1.0       | > 0.6: Tutarlı sistem ✅ <br> 0.4 - 0.6: Geliştirmeye açık ⚠️ <br> < 0.4: Model başarısız ❌         |
| Log Loss               | Uçtan uca sistemin belirsizlik derecesi                              | 0.0 - ∞          | < 0.5: Net tahminler ✅ <br> 0.5 - 1.0: Orta seviye ⚠️ <br> > 1.0: Kararsız model ❌                 |
| Classification Report  | Tüm sınıflar özelinde detaylı değerlendirme                         | 0.0 - 1.0        | > 0.8: Dengeli tahminler ✅ <br> 0.6 - 0.8: Gözden geçirilmeli ⚠️ <br> < 0.6: Zayıf sonuçlar ❌      |


| Test Adı               | Kullanım Amacı                                                        | Değer Aralığı    | Matematiksel İfade                                                                                       | Anlamı & Aksiyon                                                                                   |
|------------------------|------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Accuracy               | Tüm doğru tahminlerin oranı                                           | 0.0 - 1.0        | (TP + TN) / (TP + TN + FP + FN)                                                                           | > 0.8: Yüksek doğruluk ✅ <br> < 0.6: Model başarısız ❌                                             |
| Balanced Accuracy      | Her sınıfın doğru tahmin oranlarının ortalaması                      | 0.0 - 1.0        | (Recall₁ + Recall₂) / 2                                                                                   | Dengesiz veri setlerinde daha adil ölçüm sağlar.                                                  |
| Precision              | Pozitif tahminlerin ne kadarı gerçekten pozitif                      | 0.0 - 1.0        | TP / (TP + FP)                                                                                            | Yüksek: Az yanlış alarm ✅ <br> Düşük: Çok fazla yanlış pozitif ❌                                  |
| Recall (Sensitivity)   | Gerçek pozitiflerin ne kadarı doğru tahmin edilmiş                   | 0.0 - 1.0        | TP / (TP + FN)                                                                                            | Yüksek: Pozitifleri kaçırmıyor ✅ <br> Düşük: Pozitif kaçırılıyor ❌                               |
| F1 Score               | Precision & Recall’in harmonik ortalaması                            | 0.0 - 1.0        | 2 * (Precision * Recall) / (Precision + Recall)                                                           | Dengesizlik varsa kullanılır. Dengeli yüksek skor idealdir.                                       |
| ROC-AUC Score          | ROC eğrisinin altındaki alan                                          | 0.5 - 1.0        | AUC = ∫ TPR(FPR) dFPR                                                                                     | 0.5: Rastgele, 1.0: Mükemmel ayrım ✅                                                               |
| Gini Coefficient       | ROC eğrisine dayalı ayrım ölçüsü                                     | 0.0 - 1.0        | Gini = 2 * AUC - 1                                                                                        | Yüksek Gini, daha iyi ayrım gücü demektir.                                                        |
| Matthews CorrCoef (MCC)| Doğruluk ve dengesizlik durumlarını kapsayan korelasyon ölçüsü       | -1.0 - 1.0       | (TP*TN - FP*FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))                                                         | Dengesiz veri setlerinde daha güvenilir bir ölçüdür.                                              |
| Jaccard Score          | Gerçek ve tahmin kesişiminin birleşime oranı                         | 0.0 - 1.0        | TP / (TP + FP + FN)                                                                                       | Yüksek: Tahmin ile gerçekler örtüşüyor ✅                                                          |
| Hamming Loss     | Tahmin edilen etiketler ile gerçek etiketler arasındaki hata oranı             | 0.0 - 1.0        | (1/n_samples) * ∑(yᵢ ≠ ŷᵢ)                                                                                                 | Ne kadar küçükse o kadar iyi ✅ <br> Çoklu etiket sınıflandırmalarda özellikle önemlidir.                |
| Cohen’s Kappa    | Gerçek uyum ile beklenen uyum arasındaki fark (şansa göre düzeltilmiş doğruluk) | -1.0 - 1.0       | (Po - Pe) / (1 - Pe)                                                                                                       | > 0.8: Mükemmel uyum ✅ <br> < 0.4: Zayıf uyum ❌ <br> Özellikle gözlemciler arası uyum için kullanılır.  |
| Log Loss         | Olasılıksal sınıflandırma hatasını ölçer                                        | 0.0 - ∞          | - (1/n) * ∑[yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ)]                                                                             | Düşük değerler daha iyi. Modelin olasılık tahminlerinin ne kadar güvenilir olduğunu ölçer.             |
| Confusion Matrix       | Tahminlerin doğru/yanlışlarını tablo halinde sunar                   | -                | \[[TP, FP], [FN, TN]\]                                                                                      | Modelin hangi sınıfı karıştırdığını doğrudan gösterir.                                            |
