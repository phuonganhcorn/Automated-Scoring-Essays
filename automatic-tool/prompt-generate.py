  GNU nano 7.2                                                                                 generate_prompt.py                                                                                           
from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
from multiprocessing import Pool
import tqdm
from selenium.webdriver.support.ui import Selectimport json



nb_processes = 16
max_waiting_time = 1200 # 120s
# Providers: Auto, You, Bing, HuggingChat, Gemini, Phind, ...
# Models: Default, gpt-3.5-turbo, gemini-pro, ....
#setting = {"provider": "You", "model": "gpt-4"}
#setting = {"provider": "Auto", "model": "Default"}
setting = {"provider": "Bing", "model": "Default"}

json_files = glob.glob("/home/phanh/Downloads/demo_aap_data/*")
def extract_prompt_essay_from_json(json_files):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    prompt = data.get('question', '')
    essay = data.get('essay', '')
    
    return prompt, essay
  
promps_list = []
essay_list = []
for json_file in json_files:
    prompt, essay = extract_prompt_essay_from_json(json_file)
    prompt_list.append(prompt)
    essay_list.append(essay)
    #print("Prompt:", prompt)
    #print("Essay:", essay)
    


def generate_prompt(prompt, essay):
    try:
        prompt_template = """We have 7 Level of requirement satisfaction as below:
        1. The answer is slightly relevant to the topic, but it does not really address
        any part of the question
        2. The answer addresses a very small part of the question
        3. The answer only addresses part of the question
        The answer has addressed a part (less than 50%) of the task’s requirement
        4. Most parts of the questions are addressed, but some parts of the answer
        lack focus. The answer has addressed a bigger part (50-70%) of the task’s
        requirement5. All parts are addressed but some parts are answered better than others.
        The answer has managed to address all parts of the task’s requirement
        but has yet covered all the points thoroughly
        6. All parts are addressed, most parts are answered in-depth
        7. The answer is so full that the marker has nothing to add
        The answer has managed to address all parts of the task’s requirement
        and points relating to advantages and disadvantages have been covered
        thoroughly
        I have this prompt: {prompt}
        And the essay for upper prompt: {essay}
        Give me the score for this essay. Only the number score."""
        
        template_integrated = prompt_template.format(prompt=prompt, essay=essay)

        driver = webdriver.Firefox()
        driver.get('http://127.0.0.1:8080/chat/')
        sleep(1)

        if setting["provider"] != "Auto":
            select_provider = Select(driver.find_element(By.ID, 'provider'))
            select_provider.select_by_visible_text(setting["provider"])
            sleep(1)

        # fill text
        input_element = driver.find_element(By.ID, "message-input")
        input_element.send_keys(gaml_code_template_integrated)
        sleep(2)

        check_successful_click = False

        while check_successful_click == False:
            check1 = driver.find_element(By.CLASS_NAME, "stop_generating-hidden")
            check2 = driver.find_element(By.CLASS_NAME, "regenerate-hidden")
            
            if check1 and check2:
                input_element = driver.find_element(By.ID, "send-button")
                input_element.click()
                sleep(1)
                check_successful_click = True

        # wait maximum 100s or until the element is clickable
        element = WebDriverWait(driver, max_waiting_time).until(EC.element_to_be_clickable((By.ID, "regenerateButton")))

        output_element = driver.find_element(By.CLASS_NAME, "content_inner")
        # get prompt in html
        prompt = output_element.get_attribute('innerHTML')

        driver.find_element(By.XPATH, "//button[@onclick=\"delete_conversations()\"]").click()

        sleep(1)

        driver.quit()

        return {"question":prompt, "answer": essay}
    
    except Exception as e:
        print(e)
        return ''

for prompt, essay in zip(prompts_list, essays_list):
    generate_prompt(prompt, essay)
file_name = '/home/phanh/Downloads/test/data/train/json/test.json'
with open(file_name) as f:
    data = json.load(f)
    data_list = data["data"]["prompts"]
    final_data = [gaml_code["answer"] for d in data_list]

print(generate_prompt(final_data[0]))
