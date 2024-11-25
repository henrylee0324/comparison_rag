import os
import json
import time
import numpy as np
from retrying import retry
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

class Etl:
    def __init__(self, 
                 documents_address = "documents",
                 glossary_address = "glossary",
                 embedding_address = "embeddings",
                 chunk_size = 400,
                 overlap = 150, 
                 use_summary = True, 
                 model_name = llm_model,
                 embedding_model = embedding_model, 
                 api_key = api_key):
        self.documents_address = documents_address
        self.glossary_address = glossary_address
        self.embedding_address = embedding_address
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_summary = use_summary
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.embedding_id = 0
        os.makedirs(f"{self.embedding_address}", exist_ok=True)

    def create_document_embeddings(self):
        print("Creating document embeddings...")
        self._create_embeddings(self.documents_address, self.documents_address)
        print("Finish creating document embeddings.")
    def create_glossary_embeddings(self):
        print("Creating glossary embeddings...")
        self._create_embeddings(self.glossary_address, self.glossary_address)
        print("Finish creating glossary embeddings.")



    def _create_embeddings(self, output_name, input_address):
        txt_files = [f for f in os.listdir(input_address) if f.endswith('.txt') and os.path.isfile(os.path.join(input_address, f))]
        chunks = []
        for txt_file in txt_files:
            file_path = os.path.join(input_address, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                metadatas = self._chunking(content, txt_file)
                for metadata in metadatas:
                    summary = self._get_summary(metadata["content"])
                    content_with_summary = f"summary:\n summary \n\n content:\n {metadata['content']}"
                    embedding = self._get_embedding(content_with_summary)
                    chunks.append({
                        "embedding_id": self.embedding_id,
                        "summary": summary,
                        "embedding": embedding,
                        "metadata": metadata
                    })
                    self.embedding_id+=1
                    print(f"{self.embedding_id} finished")
                print(f"{txt_file} finsished")
        embedding_file = os.path.join(self.embedding_address, f"{output_name}.json")
        print(f"Creating embedding file at...{embedding_file}")
        with open(embedding_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {embedding_file}")           

    def _chunking(self, document, document_name):
        chunks = []
        chunk_id = 0
        document_length = len(document)
        start = 0
        while start < document_length:
            end = min(start+self.chunk_size, document_length)
            chunk = document[start:end]
            chunks.append({
                "document_name": document_name,
                "content": chunk,
                "chunk_id": chunk_id,
                "start": start
            })
            chunk_id += 1
            start = start+self.chunk_size-self.overlap
        
        return chunks

    def _get_summary(self, text):
        summary_prompt = f"""Give me the summary of following text:

        {text}

        summary:
        """
        summary = self._call_llm("", summary_prompt)
        return summary
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _get_embedding(self, text):
        client = OpenAI(api_key=self.api_key)
        text = text.replace("\n", " ")
        try:
            embedding = client.embeddings.create(input = [text], model=self.embedding_model).data[0].embedding
            # Extract the embedding from the response
            return embedding
        except Exception as e:  # 處理其他潛在錯誤
            print(f"Unexpected Error: {e}")
            raise


    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _call_llm(self, system_prompt, user_prompt):
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
        except Exception as e:  # 處理其他潛在錯誤
            print(f"Unexpected Error: {e}")
            raise
    
if __name__ == "__main__":
    etl = Etl()
    #etl.create_glossary_embeddings()
    etl.create_document_embeddings()



    
"""
{
embedding_id:
summary,
embedding:,
metadata:
    {
    document_name:
    content:,
    chunk_id:,
    start,
    }
}
"""
