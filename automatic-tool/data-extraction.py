import glob
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

train_sents=[]
test_sents=[]

stop_words = set(stopwords.words('english'))

def sentence_to_words(sentence):
    sentence = re.sub("[^A-Za-z]"," ", sentence)
    sentence = sentence.lower()
    filtered_sentence = [word for word in sentence.split() if word not in stop_words]
    return filtered_sentence

def essay_to_words(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(essay)
    final_words = [sentence_to_words(sentence) for sentence in sentences if len(sentence) > 0]
    return final_words

for essay in train_e:
    train_sents += essay_to_words(essay)

for essay in test_e:
    test_sents += essay_to_words(essay)

# Saving processed essays to files
with open("train_processed.txt", "w") as train_file:
    for sentence in train_sents:
        train_file.write(" ".join(sentence) + "\n")

with open("test_processed.txt", "w") as test_file:
    for sentence in test_sents:
        test_file.write(" ".join(sentence) + "\n")

