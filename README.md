
# Hybrid URL Classification System (ML + RL)

This project implements a two-stage hybrid system for URL classification. It utilizes both traditional Machine Learning (ML) models and a Deep Reinforcement Learning (DRL) agent for binary classification (benign vs. malicious). If a URL is identified as malicious, another DRL agent determines its specific threat type (e.g., phishing, malware, defacement, spam).

## Key Features

- **Modular Model Training**: Each classifier (RF, SVM, XGBoost, DQN) is trained with its own dedicated script, providing flexibility and ease of management.
- **Two-Stage Hybrid Architecture**:
  1.  **Binary Classifier**: Determines if a URL is `benign` or `malicious`. Four different models are trained for this stage.
  2.  **Multi-Class Classifier**: For URLs labeled as malicious, a Deep Q-Network (DQN) agent classifies the specific type of threat.
- **Comprehensive Evaluation**: Includes a dedicated script for end-to-end testing of the hybrid system, allowing for performance comparison of different binary classifier combinations.

## Project Structure

The project has a modular structure that clearly separates the responsibilities of each component:

```
.
├── data/
│   └── cleaned_feature_data.csv        # Dataset for training and testing
│
├── binary-model/                       # Training scripts for BINARY classifiers
│   ├── dqn-binary/
│   │   ├── dqn_binary.py               #   - Binary DQN training script
│   │   └── env_url_type.py             #   - Gym environment for Binary DQN
│   ├── rf_binary.py                    #   - Random Forest training script
│   ├── svm_binary.py                   #   - SVM training script
│   └── xgb_binary.py                   #   - XGBoost training script
│
├── binary-model-save/                  # Directory for saved TRAINED BINARY models
│   ├── binary_dqn_model.zip
│   ├── rf_binary_model.pkl
│   ├── svm_binary_model.pkl
│   └── xgb_binary_model.pkl
│
├── multiclass-model/                   # Training scripts for the MULTI-CLASS classifier
│   ├── dqn_model.py                    #   - Multiclass DQN training script
│   └── env_url_type.py                 #   - Gym environment for Multiclass DQN
│
├── multiclass-model-save/              # Directory for the saved TRAINED MULTI-CLASS model
│   └── multiclass_dqn_model.zip
│
├── test/
│   └── system_test.py                  # End-to-end test script for the hybrid system
│
├── install_dependencies.py             # Helper script to install required libraries
├── requirements.txt                    # Project dependencies list
├── README.md
└── README_tr.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Run the provided installation script:
    ```bash
    python install_dependencies.py
    ```
    Alternatively, you can install directly from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Running the project consists of three main steps: training the binary models, training the multi-class model, and finally, testing the system as a whole.

### Step 1: Train the Binary Classifiers

In this stage, you can train the four different binary classifiers by running each script in the `binary-model/` directory individually. Each script will save its trained model to the `binary-model-save/` directory.

```bash
# To train the Random Forest model:
python binary-model/rf_binary.py

# To train the SVM model:
python binary-model/svm_binary.py

# To train the XGBoost model:
python binary-model/xgb_binary.py

# To train the Binary DQN model:
python binary-model/dqn-binary/dqn_binary.py
```

### Step 2: Train the Multi-Class Classifier

In this step, the DQN agent that will classify the malicious URL types is trained.

```bash
# To train the Multiclass DQN model:
python multiclass-model/dqn_model.py
```
This command will save the trained model to the `multiclass-model-save/` directory.

### Step 3: Evaluate the Hybrid System

After all models are trained, you can test the end-to-end performance of the system using the `test/system_test.py` script. This script allows you to choose which binary classifier you want to use for the first stage.

**Example Usages:**

*   To test the **Random Forest + DQN** hybrid structure:
    ```bash
    python test/system_test.py --binary_model rf
    ```

*   To test the **Binary DQN + Multiclass DQN** hybrid structure:
    ```bash
    python test/system_test.py --binary_model dqn
    ```

*   To test the **XGBoost + DQN** hybrid structure:
    ```bash
    python test/system_test.py --binary_model xgb
    ```

These commands will load the selected binary classifier and the multi-class DQN model to perform a full performance analysis and display the results (metrics, confusion matrices).