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
from selenium.webdriver.support.ui import Select


nb_processes = 2
max_waiting_time = 1500 # 120s
# Providers: Auto, You, Bing, HuggingChat, Gemini, Phind, ...
# Models: Default, gpt-3.5-turbo, gpt-4, gemini-pro, ....
#setting = {"provider": "You", "model": "gpt-4"} # 20 - 30% success
#setting = {"provider": "You", "model": "Default"} # 60% success
setting = {"provider": "Auto", "model": "Default"} # > 80 % success
#setting = {"provider": "Bing", "model": "gpt-4"} # > 80 % success

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

        if setting["model"] != "Default":
            select_model = Select(driver.find_element(By.ID, 'model'))
            select_model.select_by_visible_text(setting["model"])
            sleep(1)

