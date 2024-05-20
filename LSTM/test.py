import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.models import load_model
from model import sent2word, essay2word, get_vecs  # Import necessary functions from model.py

# Ensure required NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_new_essay(essay, stop_words, model, num_features):
    essay_sents = essay2word(essay, stop_words)
    essay_vecs = get_vecs([essay_sents], model, num_features)
    essay_vecs = np.reshape(essay_vecs, (essay_vecs.shape[0], 1, essay_vecs.shape[1]))
    return essay_vecs

# Main script for inference
if __name__ == "__main__":
    # Paths to model files
    word2vec_model_path = 'word2vecmodel.bin'
    lstm_model_path = 'final_lstm.h5'

    # Stop words set
    stop_words = set(stopwords.words('english'))

    # Load trained models
    word2vec_model = Word2Vec.load(word2vec_model_path)
    lstm_model = load_model(lstm_model_path)

    # Example new essay for inference
    new_essay = str(input("Your example essay text goes here. This should be replaced with the actual essay text for inference:\n"))

    # Preprocess new essay
    new_essay_vector = preprocess_new_essay(new_essay, stop_words, word2vec_model, word2vec_model.vector_size)

    # Predict score for new essay
    y_pred = lstm_model.predict(new_essay_vector)
    y_pred = np.around(y_pred)
    print(f'Predicted score: {y_pred[0][0]}')

