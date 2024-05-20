import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.metrics import mean_squared_error, cohen_kappa_score

# Ensure required NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def load_and_process_data(training_set_path, processed_data_path):
    # Load data
    df = pd.read_csv(training_set_path, sep='\t', encoding='ISO-8859-1')
    df.dropna(axis=1, inplace=True)
    df.drop(columns=['domain1_score', 'rater1_domain1', 'rater2_domain1'], inplace=True, axis=1)

    # Load processed scores
    temp = pd.read_csv(processed_data_path)
    temp.drop("Unnamed: 0", inplace=True, axis=1)
    temp.reset_index(drop=True, inplace=True)
    df['domain1_score'] = temp['final_score']

    # Split data into features and target
    y = df['domain1_score']
    df.drop('domain1_score', inplace=True, axis=1)
    X = df

    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_e = X_train['essay'].tolist()
    test_e = X_test['essay'].tolist()
    return train_e, test_e, y_train, y_test

def sent2word(x, stop_words):
    x = re.sub("[^A-Za-z]", " ", x).lower()
    filtered_sentence = [w for w in x.split() if w not in stop_words]
    return filtered_sentence

def essay2word(essay, stop_words):
    essay = essay.strip()
    tokenizer = nltk.data.load('./tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = [sent2word(sent, stop_words) for sent in raw if len(sent) > 0]
    return final_words

def process_essays(essays, stop_words):
    sentences = []
    for essay in essays:
        sentences += essay2word(essay, stop_words)
    return sentences

def train_word2vec(sentences, vector_size=300, min_count=40, workers=4, window=10, sample=1e-3):
    model = Word2Vec(sentences,
                     workers=workers,
                     vector_size=vector_size,
                     min_count=min_count,
                     window=window,
                     sample=sample)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format('./word2vecmodel.bin', binary=True)
    return model

def make_vec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    no_of_words = 0.
    for word in words:
        if word in model.wv.key_to_index:
            no_of_words += 1
            vec = np.add(vec, model.wv[word])
    vec = np.divide(vec, no_of_words)
    return vec

def get_vecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for idx, essay in enumerate(essays):
        essay_vecs[idx] = make_vec(essay, model, num_features)
    return essay_vecs

def get_model(input_shape=(1, 300)):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model

def train_and_evaluate_model(training_vectors, y_train, testing_vectors, y_test):
    lstm_model = get_model(input_shape=(1, training_vectors.shape[2]))
    lstm_model.fit(training_vectors, y_train, batch_size=64, epochs=150)
    lstm_model.save('final_lstm.h5')

    predictions = lstm_model.predict(testing_vectors)
    mse = mean_squared_error(y_test, predictions)
    kappa = cohen_kappa_score(y_test, np.round(predictions))
    print(f'Mean Squared Error: {mse}')
    print(f'Cohen Kappa Score: {kappa}')
    return lstm_model

# Main script
if __name__ == "__main__":
    # Paths to data files
    training_set_path = "./Data/training_set_rel3.tsv"
    processed_data_path = "./Data/Processed_data.csv"

    # Load and process data
    X, y = load_and_process_data(training_set_path, processed_data_path)
    train_e, test_e, y_train, y_test = split_data(X, y)

    # Stop words set
    stop_words = set(stopwords.words('english'))

    # Process essays into sentences
    train_sents = process_essays(train_e, stop_words)
    test_sents = process_essays(test_e, stop_words)

    print(f"Number of training sentences: {len(train_sents)}")
    print(f"Number of testing sentences: {len(test_sents)}")

    # Train Word2Vec model
    word2vec_model = train_word2vec(train_sents)

    # Prepare training and testing vectors
    clean_train = [sent2word(essay, stop_words) for essay in train_e]
    training_vectors = get_vecs(clean_train, word2vec_model, word2vec_model.vector_size)

    clean_test = [sent2word(essay, stop_words) for essay in test_e]
    testing_vectors = get_vecs(clean_test, word2vec_model, word2vec_model.vector_size)

    training_vectors = np.array(training_vectors)
    testing_vectors = np.array(testing_vectors)

    # Reshape vectors for LSTM input
    training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
    testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))

    # Train and evaluate the LSTM model
    train_and_evaluate_model(training_vectors, y_train, testing_vectors, y_test)

