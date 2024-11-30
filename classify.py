import os
import json
import shutil
from dotenv import load_dotenv
from openai import OpenAI
import time
from datetime import datetime


load_dotenv()

api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL")

class Classifier:
    def __init__(self, condition, 
                 input_address="documents", 
                 group_address = "group",
                 model_name=llm_model, 
                 api_key=api_key, 
                 retry_attempts=3, 
                 retry_delay=2):
        self.model_name = model_name
        self.condition = condition
        self.input_address = input_address
        self.group_address = group_address
        self.api_key = api_key
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.group_name_1 = ""
        self.group_name_2 = ""
        self.group_1 = []
        self.group_2 = []
        self.group_file = ""
        os.makedirs(f"{self.group_address}", exist_ok=True)
        self._get_group_names()
        self.classify_documents()
    
    def get_group_name(self):
        return self.group_name_1, self.group_name_2
    
    def classify_documents(self):
        classify_prompt = f"""I will provide you with an article and two group names: "{self.group_name_1}" and "{self.group_name_2}". 
        Based on the content of the article, determine which group the article most likely belongs to.
        Your response should only contains the name of group.
        """
        try:
            txt_files = [f for f in os.listdir(self.input_address) if f.endswith('.txt') and os.path.isfile(os.path.join(self.input_address, f))]
            for txt_file in txt_files:
                file_path = os.path.join(self.input_address, txt_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        group_name = self._call_llm(classify_prompt, content)
                        print(group_name)
                        if group_name == self.group_name_1:
                            self.group_1.append(txt_file)
                        elif group_name == self.group_name_2:
                            self.group_2.append(txt_file)
                        else:
                            print(f"Group {group_name} does not exist.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        except Exception as e:
            print(f"Error accessing input folder {self.input_address}: {e}")
        group_log = {
            "group_1": {
                "group_name": self.group_name_1,
                "documents": self.group_1
            },
            "group_2": {
                "group_name": self.group_name_2,
                "documents": self.group_2
            }
        }
        self.group_file = os.path.join(self.group_address, f"{datetime.now().strftime("%Y%m%d_%H%M%S") }.json")
        with open(self.group_file, "w", encoding="utf-8") as f:
            json.dump(group_log, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {self.group_file}")

        return self.group_file

    def _get_group_names(self):
        dict = """
        {
        "group1":
        "group2":
        }
        """
        groupname_prompt = f"""According to "{self.condition}", you need to divide documents into two groups.
        Please give me the names of two groups as following format:

        {dict}

        Your answer should contain only two group names.
        """
        try:
            res = self._call_llm("", groupname_prompt)
            res = json.loads(res)
            print(f"Response:\n{res}")
            self.group_name_1 = res["group1"]
            self.group_name_2 = res["group2"]
            print("Group names created.")
            print(f"Group1 name: {self.group_name_1}")
            print(f"Group2 name: {self.group_name_2}")
        except Exception as e:
            print(f"Error getting group names: {e}")
        
        return self.group_name_1, self.group_name_2

    def _call_llm(self, system_prompt, user_prompt):
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                client = OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                attempts += 1
                print(f"Error calling LLM API (Attempt {attempts}/{self.retry_attempts}): {e}")
                if attempts < self.retry_attempts:
                    time.sleep(self.retry_delay)  # 延遲後重試
                else:
                    raise RuntimeError(f"Failed to call LLM API after {self.retry_attempts} attempts.") from e

if __name__ == "__main__":
    print("Please enter your condition to divide documents into two groups.")
    condition = input("Enter the condition to divide documents into two groups: ").strip()
    print("Enter the folder address where your documents are in.")
    input_address = input("Input folder address: ").strip()
    classifier = Classifier(condition, input_address=input_address)
