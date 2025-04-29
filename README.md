# SMS Spam Detection with PyTorch  

A feedforward neural network to classify SMS messages as **Spam** or **Ham** (non-spam) using PyTorch and TF-IDF features.

## 🚀 Project Overview
This repository implements an SMS spam classifier:
- **Preprocessing**: clean messages (remove URLs, emails, numbers, punctuation).  
- **Feature Extraction**: TF-IDF vectorization (unigrams, English stopwords).  
- **Model**: PyTorch feedforward neural network.  
- **Evaluation**: accuracy, precision, recall & F1-score on training & test sets.  

---

## ✨ Features
- **Text Cleaning**: lowercase, strip unwanted tokens, normalize whitespace  
- **TF-IDF Vectorization**: max_df=0.9, min_df=5, removes very common/rare terms  
- **Neural Network**:  
  - 1 hidden layer (100 neurons, ReLU)  
  - Output layer for binary classification  
- **Metrics**: classification report + individual scores  

---

## 🛠️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/TheRealSoutrikDas/Spam-SMS-Detection.git
   cd Spam-SMS-Detection
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**  
   Place `spam.csv` in the project root (download from Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 🎯 Usage

### 1. Training
```bash
python main.py
```
- Trains the model for 10 epochs  
- Saves:
  - `spam_classifier.pt` (model weights)  
  - `tfidf_vectorizer.pkl` (fitted TF-IDF object)  

### 2. Prediction
```bash
python spam_pred_script.py
```
- Prompts for an SMS message  
- Outputs **Spam** or **Ham**

---

## 🧩 Model Architecture
```
Input (TF-IDF vector size) 
   ↓
Linear (input_dim → 100) 
   ↓
ReLU 
   ↓
Linear (100 → 2) 
   ↓
Softmax (inside CrossEntropyLoss)
```

- Loss: `CrossEntropyLoss`  
- Optimizer: `Adam` (lr=1e-3)

---

## 📊 Evaluation Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)  
- **Precision**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)  

Metrics are reported on both **training** and **test** sets.

---

## 📈 Results

| Dataset      | Accuracy | Precision | Recall | F1-Score |
|------------: |:--------:|:---------:|:------:|:--------:|
| **Training** | 0.9987   | 1.00      | 0.9904 | 0.9952   |
| **Test**     | 0.9797   | 0.9318    | 0.9152 | 0.9234   |

*Values may vary slightly per run.*

---

## 🔮 Future Improvements
- **Hyperparameter Tuning**: adjust layers, learning rate, batch size  
- **Regularization**: add dropout, weight decay  
- **Advanced Models**: CNNs/RNNs over embeddings or transformer-based (BERT)  
- **Class Imbalance**: weighted loss or oversampling  
- **Cross-Validation**: k-fold for robust evaluation  

---

## 🤝 Acknowledgements
- **Dataset**: SMS Spam Collection Dataset ([link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset))  
- **Libraries**: PyTorch, scikit-learn, pandas, joblib  

---

## 📝 License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
