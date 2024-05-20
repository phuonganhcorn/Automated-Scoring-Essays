import glob
import json

json_files = glob.glob("/home/phanh/Downloads/demo_aap_data/*")

def read_json(json_files):
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            content = json.load(f)
            data.append(content)
    return data
data = read_json(json_files)
print(len(data))


def extract_unique_question_ids(data):
    question_ids = set()  
    question = set()
    for entry in data:
        question_ids.add(entry['question_id'])
        question.add(entry['question'])
    return list(question_ids), list(question)
question_ids, questions = extract_unique_question_ids(data)
question_ids = sorted(question_ids)
print(question_ids)

def extract_answer_with_question_ids(data, question_ids):
    essays_by_question_id = {q_id: [] for q_id in question_ids}
    
    for entry in data:
        q_id = entry['question_id']
        if q_id in essays_by_question_id:
            essays_by_question_id[q_id].append({'question': entry['question'], 'essay': entry['essay']})
    
    for q_id, essays in essays_by_question_id.items():
        filename = f"./data/question-{q_id}.json"
        with open(filename, 'w') as json_file:
            json.dump(essays, json_file, indent=4)
    
    return essays_by_question_id    
                
#extract_answer_with_question_ids(data, question_ids)