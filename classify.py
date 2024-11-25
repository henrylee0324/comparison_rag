import os
import json
import shutil
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL")

class Classifier:
    def __init__(self, condition, 
                 input_address = "documents", 
                 output_address = "classified_documents", 
                 model_name = llm_model, 
                 api_key = api_key):
        self.model_name = model_name
        self.condition = condition
        self.input_address = input_address
        self.output_address = output_address
        os.makedirs(f"{self.output_address}", exist_ok=True)
        print(f"Output folder '{self.output_address}' created.")
        self.api_key = api_key
        self.group_name_1 = ""
        self.group_name_2 = ""
        self._get_group_names()
        self._move_documents()
    
    def get_group_name(self):
        return self.group_name_1, self.group_name_2
    
    def _move_documents(self):
        classify_prompt = f"""I will provide you with an article and two group names: "{self.group_name_1}" and "{self.group_name_2}". 
        Based on the content of the article, determine which group the article most likely belongs to.
        Your response should only contains the name of group.
        """
        txt_files = [f for f in os.listdir(self.input_address) if f.endswith('.txt') and os.path.isfile(os.path.join(self.input_address, f))]
        for txt_file in txt_files:
            file_path = os.path.join(self.input_address, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                group_name = self._call_llm(classify_prompt, content)
                print(group_name)
                group_folder = f"{self.output_address}/{group_name}/{txt_file}"
                shutil.copy2(file_path, group_folder)

    def _get_group_names(self):
        dict = """
        {
        "group1":
        "group2":
        }
        """
        groupname_prompt = f"""Accourding to "{self.condition}", you need to divid documents into two groups.
        Please give me the names of two group as following format:

        {dict}

        Your answer should contain only two groups name.
        """
        res = self._call_llm("", groupname_prompt)
        res = json.loads(res)
        self.group_name_1 = res["group1"]
        self.group_name_2 = res["group2"]
        print("Group name created.")
        print(f"group1 name: {self.group_name_1}")
        print(f"group2 name: {self.group_name_2}")
        os.makedirs(f"{self.output_address}/{self.group_name_1}", exist_ok=True)
        os.makedirs(f"{self.output_address}/{self.group_name_2}", exist_ok=True)
        print("Folder created")

        return self.group_name_1, self.group_name_2

    def _call_llm(self, system_prompt, user_prompt):
        client = OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        )

        return completion.choices[0].message.content


if __name__ == "__main__":
    print("Please enter your condition to divid documents into two groups.")
    condition = input("Enter the condition to divid documents into two groups:").strip()
    print("Enter the folder address where your documents are in.")
    input_address = input("Input folder address:").strip()
    print("Enter the output folder address.")
    output_address = input("Output folder address:").strip()
    classifier = Classifier(condition, input_address=input_address, output_address=output_address)
