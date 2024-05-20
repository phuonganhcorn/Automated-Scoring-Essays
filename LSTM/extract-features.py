import glob
import json
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import csv

nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def read_json(json_files):
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            content = json.load(f)
            data.append(content)
    return data

json_files = glob.glob("/home/phanh/Downloads/Automated-Scoring-Essays/test-data/demo_aap_data/*.json")

data_list = read_json(json_files)

def process_essay(data):
    essay = data["essay"]
    clean_essay = re.sub(r'\n', ' ', essay)
    char_count = len(clean_essay)
    words = word_tokenize(clean_essay)
    word_count = len(words)
    sentences = sent_tokenize(clean_essay)
    sent_count = len(sentences)
    avg_word_len = sum(len(word) for word in words) / word_count
    dictionary = set(nltk.corpus.words.words())
    spell_err_count = sum(1 for word in words if word.lower() not in dictionary)
    pos_tags = nltk.pos_tag(words)
    noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
    adj_count = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
    verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
    adv_count = sum(1 for word, pos in pos_tags if pos.startswith('RB'))
    final_score = 0
    extracted_data = {
        "essay_id": data["essay_id"],
        "question_id": data["question_id"],
        "question": data["question"],
        "essay": data["essay"],
        "final_score": final_score,
        "clean_essay": clean_essay,
        "char_count": char_count,
        "word_count": word_count,
        "sent_count": sent_count,
        "avg_word_len": avg_word_len,
        "spell_err_count": spell_err_count,
        "noun_count": noun_count,
        "adj_count": adj_count,
        "verb_count": verb_count,
        "adv_count": adv_count
    }
    return extracted_data

extracted_data_list = [process_essay(data) for data in data_list]

csv_file = "extracted_data.csv"

with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = [
        "essay_id", "question_id", "question", "essay", "final_score", 
        "clean_essay", "char_count", "word_count", "sent_count", 
        "avg_word_len", "spell_err_count", "noun_count", 
        "adj_count", "verb_count", "adv_count"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in extracted_data_list:
        writer.writerow(data)

print(f"Extracted data has been written to {csv_file}")

