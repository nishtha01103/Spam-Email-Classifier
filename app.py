import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources (do only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.data.path.append('./nltk_data')

# Initialize stemmer
ps = PorterStemmer()


# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
tfidf = joblib.load(open('vectorizer.pkl', 'rb'))
model = joblib.load(open('model.pkl', 'rb'))

# Streamlit app UI
st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
    }
    .stApp {
        background-image: linear-gradient(145deg, #ffffff, #e0e0e0);
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        text-align: center;
        color: #4a4a4a;
        font-size: 42px;
        font-weight: 700;
        padding-top: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777777;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    .css-1v0mbdj.e115fcil2 {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 16px;
        border: none;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.markdown('<div class="title">📩 Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter an email or SMS message to detect whether it\'s spam or not</div>', unsafe_allow_html=True)

input_sms = st.text_input("🔍 Enter your message below:")

if st.button('🚀 Predict'):
    # Step 1: Preprocess
    transformed_sms = transform_text(input_sms)

    # Step 2: Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Step 3: Predict
    result = model.predict(vector_input)[0]

    # Step 4: Display Result
    if result == 1:
        st.markdown(
            "<div style='color: black; font-size: 20px; font-weight: bold;'>❌ This message is <span style='color: red;'>Spam</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='color: black; font-size: 20px; font-weight: bold;'>✅ This message is <span style='color: green;'>Not Spam</span></div>",
            unsafe_allow_html=True
        )



