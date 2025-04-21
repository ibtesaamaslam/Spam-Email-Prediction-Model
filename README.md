
## 📧 Spam Mail Prediction Model

This project focuses on building a **machine learning-based model to detect spam emails** with high accuracy. By leveraging natural language processing (NLP) and supervised learning techniques, this model classifies emails as either **Spam** or **Ham (Not Spam)** based on their content. This can be integrated into real-world applications like email clients, messaging platforms, or cybersecurity systems to improve user safety and filter unwanted messages.

---

### 🚀 Project Highlights

- **Dataset Used**: The project utilizes the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — a well-known corpus with 5,574 messages labeled as spam or ham.
- **Tech Stack**:
  - Language: Python
  - Environment: Jupyter Notebook
  - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `nltk`

---

### 🧠 Machine Learning Pipeline

#### 1. **Data Preprocessing**
- Handled missing/null values.
- Converted categorical labels ('spam'/'ham') into binary values (1/0).
- Performed EDA (Exploratory Data Analysis) with visualizations for message lengths, word counts, etc.

#### 2. **Text Processing with NLP**
- Cleaned text using:
  - Lowercasing
  - Removal of punctuation and stopwords
  - Tokenization
  - Stemming (using NLTK's `PorterStemmer`)

#### 3. **Feature Engineering**
- Vectorization using:
  - **Bag of Words (CountVectorizer)**
  - **TF-IDF Vectorizer**
- Examined performance impact of different vectorization strategies.

#### 4. **Model Building & Evaluation**
- Implemented and evaluated multiple classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
- Best performance achieved with **Multinomial Naive Bayes + CountVectorizer**.

---

### 📊 Results

- **Accuracy Achieved**: ~98%
- **Most Efficient Model**: Naive Bayes with CountVectorizer
- Lightweight and suitable for real-time deployment.

---

### 📁 Repository Structure

```bash
Spam-Mail-Prediction-Model/
│
├── Spam Mail Prediction Model.ipynb      # Main Jupyter notebook
├── README.md                             # Project overview and instructions
├── requirements.txt                      # Dependencies
└── dataset/                              # (optional) Place for dataset files
```

---

### ✅ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/spam-mail-prediction-model.git
   cd spam-mail-prediction-model
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook "Spam Mail Prediction Model.ipynb"
   ```

---

### 📌 Future Improvements

- Add deployment with Streamlit or Flask web app
- Use deep learning models (RNN, LSTM) for advanced performance
- Implement feedback-based learning from user interaction
- Expand dataset with multilingual or multimedia messages

