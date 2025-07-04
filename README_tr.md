# Hibrit URL Sınıflandırma Sistemi (MÖ + PÖ)

Bu proje, URL sınıflandırması için iki aşamalı hibrit bir sistem uygular. Bu sistem, ikili sınıflandırma (zararsız vs. zararlı) için hem geleneksel Makine Öğrenmesi (MÖ) modellerini hem de Derin Pekiştirmeli Öğrenme (PÖ) ajanını kullanır. Eğer bir URL zararlı olarak tespit edilirse, ikinci aşamada başka bir PÖ ajanı tarafından tehdidin spesifik türü (örneğin, oltalama, kötü amaçlı yazılım, tahrifat, spam) belirlenir.

## Temel Özellikler

- **Modüler Model Eğitimi**: Her sınıflandırıcı (RF, SVM, XGBoost, DQN) kendi özel betiği ile eğitilir, bu da esneklik ve yönetim kolaylığı sağlar.
- **İki Aşamalı Hibrit Mimarisi**:
  1.  **İkili Sınıflandırıcı**: Bir URL'nin `zararsız` mı yoksa `zararlı` mı olduğunu tespit eder. Bu aşama için dört farklı model eğitilmiştir.
  2.  **Çok Sınıflı Sınıflandırıcı**: Zararlı olarak etiketlenen URL'ler için, bir Derin Q-Network (DQN) ajanı, zararlı yazılımın türünü sınıflandırır.
- **Kapsamlı Değerlendirme**: Hibrit sistemin performansını uçtan uca test etmek için özel bir betik içerir. Bu betik, farklı ikili sınıflandırıcı kombinasyonlarının sonuçlarını karşılaştırma imkanı sunar.

## Proje Yapısı

Proje, her bir bileşenin sorumluluğunu net bir şekilde ayıran modüler bir yapıya sahiptir:

```
.
├── data/
│   └── cleaned_feature_data.csv        # Eğitim ve test için kullanılan veri seti
│
├── binary-model/                       # İKİLİ sınıflandırıcıların eğitim betikleri
│   ├── dqn-binary/
│   │   ├── dqn_binary.py               #   - Binary DQN eğitim betiği
│   │   └── env_url_type.py             #   - Binary DQN için Gym ortamı
│   ├── rf_binary.py                    #   - Random Forest eğitim betiği
│   ├── svm_binary.py                   #   - SVM eğitim betiği
│   └── xgb_binary.py                   #   - XGBoost eğitim betiği
│
├── binary-model-save/                  # EĞİTİLMİŞ İKİLİ modellerin kaydedildiği yer
│   ├── binary_dqn_model.zip
│   ├── rf_binary_model.pkl
│   ├── svm_binary_model.pkl
│   └── xgb_binary_model.pkl
│
├── multiclass-model/                   # ÇOK SINIFLI sınıflandırıcının eğitim betikleri
│   ├── dqn_model.py                    #   - Multiclass DQN eğitim betiği
│   └── env_url_type.py                 #   - Multiclass DQN için Gym ortamı
│
├── multiclass-model-save/              # EĞİTİLMİŞ ÇOK SINIFLI modelin kaydedildiği yer
│   └── multiclass_dqn_model.zip
│
├── test/
│   └── system_test.py                  # Hibrit sistemin uçtan uca test betiği
│
├── install_dependencies.py             # Gerekli kütüphaneleri kuran yardımcı betik
├── requirements.txt                    # Proje bağımlılıkları listesi
├── README.md
└── README_tr.md
```

## Kurulum

1.  **Projeyi klonlayın:**
    ```bash
    git clone https://github.com/SahinMuhammetAbdullah/hybrid-url-classification.git
    cd hybrid-url-classification
    ```

2.  **Sanal ortam oluşturun (önerilir):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Bağımlılıkları yükleyin:**
    Sağlanan kurulum betiğini çalıştırın:
    ```bash
    python install_dependencies.py
    ```
    Alternatif olarak, doğrudan `requirements.txt` dosyasından yükleyebilirsiniz:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

Projenin çalıştırılması üç ana adımdan oluşur: İkili modellerin eğitimi, çok sınıflı modelin eğitimi ve son olarak sistemin bütün olarak test edilmesi.

### Adım 1: İkili Sınıflandırıcıları Eğitme

Bu aşamada, `binary-model/` klasöründeki her bir betiği ayrı ayrı çalıştırarak dört farklı ikili sınıflandırıcıyı eğitebilirsiniz. Her betik, eğittiği modeli `binary-model-save/` klasörüne kaydedecektir.

```bash
# Random Forest modelini eğitmek için:
python binary-model/rf_binary.py

# SVM modelini eğitmek için:
python binary-model/svm_binary.py

# XGBoost modelini eğitmek için:
python binary-model/xgb_binary.py

# Binary DQN modelini eğitmek için:
python binary-model/dqn-binary/dqn_binary.py
```

### Adım 2: Çok Sınıflı Sınıflandırıcıyı Eğitme

Bu adımda, zararlı URL türlerini sınıflandıracak olan DQN ajanı eğitilir.

```bash
# Multiclass DQN modelini eğitmek için:
python multiclass-model/dqn_model.py
```
Bu komut, eğitilmiş modeli `multiclass-model-save/` klasörüne kaydedecektir.

### Adım 3: Hibrit Sistemi Değerlendirme

Tüm modeller eğitildikten sonra, `test/system_test.py` betiğini kullanarak sistemin uçtan uca performansını test edebilirsiniz. Bu betik, ilk aşama için hangi ikili sınıflandırıcıyı kullanmak istediğinizi seçmenize olanak tanır.

**Örnek Kullanımlar:**

*   **Random Forest + DQN** hibrit yapısını test etmek için:
    ```bash
    python test/system_test.py --binary_model rf
    ```

*   **Binary DQN + Multiclass DQN** hibrit yapısını test etmek için:
    ```bash
    python test/system_test.py --binary_model dqn
    ```

*   **XGBoost + DQN** hibrit yapısını test etmek için:
    ```bash
    python test/system_test.py --binary_model xgb
    ```

Bu komutlar, seçilen ikili sınıflandırıcıyı ve çok sınıflı DQN modelini yükleyerek tam bir performans analizi yapar ve sonuçları (metrikler, karmaşıklık matrisleri) gösterir.
