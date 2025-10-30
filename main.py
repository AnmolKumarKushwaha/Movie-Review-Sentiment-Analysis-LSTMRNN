import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="LSTM Sentiment Analyzer", 
    initial_sidebar_state="collapsed"
)

VOCAB_SIZE = 10000 

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

try:
    @st.cache_resource
    def get_model():
        return load_model('lstmRNN.h5')
    
    model = get_model()
except OSError:
    st.error("Error: Model file 'lstmRNN.h5' not found. Please run the training script first.")
    st.stop()


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)
        
        if index >= VOCAB_SIZE:
            index = 2
            
        encoded_review.append(index + 3)
        
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    .title-wrapper {
        text-align: center;
        padding: 20px 0;
        background-color: #262730;
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .title-wrapper h1 {
        font-size: 2.5em;
        margin-bottom: 5px;
        color: #FF4B4B;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-5px);
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #155724;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #721c24;
        color: #721c24;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="title-wrapper">
        <h1>IMDB Review Sentiment Analyzer</h1>
        <p>Using an L2-regularized LSTM network to classify movie reviews.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")
st.subheader("📝 Enter Your Movie Review")

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.info("Try writing a detailed review (min 20 words) to test the model's accuracy! Be sure to use strong descriptive words to see how well the LSTM works.")
    
    user_input = st.text_area(
        'Paste your review here:', 
        height=180, 
        label_visibility="collapsed",
        placeholder="E.g., 'This film was an absolute disaster. The plot was confusing, and the acting was terrible. I would not recommend this movie to anyone.'"
    )
    
    classify_button = st.button('Classify Review', use_container_width=True)

st.write("---")

if classify_button:
    if not user_input.strip():
        st.warning('Please enter a movie review before classifying.')
    else:
        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input, verbose=0)
        score = prediction[0][0]
        
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        
        st.subheader('Classification Result')

        if sentiment == 'Positive':
            box_class = "prediction-box positive"
            conf_score = score * 100
        else:
            box_class = "prediction-box negative"
            conf_score = (1 - score) * 100

        st.markdown(
            f"""
            <div class="{box_class}">
                <h2>The Sentiment is: <span style="font-weight: bold;">{sentiment.upper()}</span></h2>
                <p style="font-size: 1.2em;">
                    Confidence: <span style="font-weight: bold; font-size: 1.5em;">{conf_score:.2f}%</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("Show Raw Prediction Details"):
            st.code(f'Raw Probability of Positive (P > 0.5): {score:.5f}')
            st.caption('The model\'s raw output is the probability that the review is positive.')
