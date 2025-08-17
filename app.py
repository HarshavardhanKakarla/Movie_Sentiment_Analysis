import streamlit as st
import pickle
import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬"
)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@st.cache_data
def load_model():
    with open('model1.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('bow.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


st.title("Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        processed_text = clean_text(user_input)
        X = vectorizer.transform([processed_text])
        prediction = model.predict(X)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(sentiment)
    else:
        st.warning("Please input a review to analyze.")
