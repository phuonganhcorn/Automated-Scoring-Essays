import gradio as gr
import numpy as np
import re
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the LSTM model
try:
    lstm_model = load_model('./Automatic-Essay-Scoring/final_lstm.h5')
except Exception as e:
    print("Error loading LSTM model:", e)

# Load the Word2Vec model
try:
    word2vec_model = KeyedVectors.load_word2vec_format('./Automatic-Essay-Scoring/word2vecmodel.bin', binary=True)
except Exception as e:
    print("Error loading Word2Vec model:", e)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Preprocess text data
    text = re.sub("[^A-Za-z]", " ", text)
    text = text.lower()
    words = text.split()
    filtered_text = [word for word in words if word not in stop_words]
    return filtered_text

def text_to_vectors(words, model, num_features=300):
    # Convert text to vectors using Word2Vec
    vec = np.zeros((num_features,), dtype="float32")
    no_of_words = 0
    for word in words:
        if word in model:
            no_of_words += 1
            vec = np.add(vec, model[word])
    if no_of_words != 0:
        vec /= no_of_words
    return vec

def predict_score(text):
    # Predict the score of the essay
    cleaned_text = preprocess_text(text)
    vectors = text_to_vectors(cleaned_text, word2vec_model)
    if vectors.any():
        padded_sequence = pad_sequences([vectors], maxlen=300, dtype='float32')
        padded_sequence = np.reshape(padded_sequence, (padded_sequence.shape[0], 1, padded_sequence.shape[1]))
        prediction = lstm_model.predict(padded_sequence)
        return float(np.around(prediction)[0][0])  # Convert prediction to a Python float
    else:
        return "Cannot make prediction for empty text."


# Create Gradio interface
iface = gr.Interface(
    fn=predict_score,
    inputs=["text", "text"],
    outputs="number",
    title="Essay Score Prediction",
    description="Enter a prompt and an essay, and get the predicted score."
)

# Launch the interface
iface.launch()
