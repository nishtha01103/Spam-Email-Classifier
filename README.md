# 📧 Spam Email/SMS Classifier

A machine learning web application that classifies messages (emails or SMS) as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.

🔗 **Live App**: [Spam Email Classifier on Render](https://spam-email-classifier-ox4v.onrender.com/)

---

## 🚀 Features

- Clean and simple Streamlit-based UI
- Preprocessing using NLTK (stopword removal, stemming)
- TF-IDF vectorization
- Naive Bayes classifier for prediction
- Visual feedback for predictions (✔️ Not Spam / ❌ Spam)

---
## 🧠 Tech Stack

- **Language**: Python
- **Libraries**:
  - Streamlit
  - Scikit-learn
  - NLTK
  - Joblib

---

## 🛠️ How It Works

1. User enters a message in the input field.
2. The message is preprocessed:
   - Lowercased
   - Tokenized
   - Stopwords and punctuation removed
   - Words stemmed using PorterStemmer
3. The cleaned message is vectorized using TF-IDF.
4. A Naive Bayes classifier predicts whether the message is spam or not.

---

## 📦 Installation (Local Setup)

```bash
# Clone the repository
git clone https://github.com/nishtha01103/Spam-Email-Classifier.git
cd Spam-Email-Classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

