import os
import json
import time
import numpy as np
from retrying import retry
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from classify import Classifier




load_dotenv()

api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

class Rag:
    def __init__(self, 
                 document_address = "documents",
                 embedding_address = "embeddings",
                 output_address = "outputs",
                 group_address = "group",
                 embedding_model = embedding_model, 
                 model_name = llm_model,
                 api_key = api_key,
                 select_chunk_number = 10,
                 cosine_similarity_threshold = 0.7):
        self.document_address = document_address
        self.embedding_address = embedding_address
        self.output_address = output_address
        self.group_address = group_address
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.api_key = api_key
        self.select_chunk_number = select_chunk_number
        self.cosine_similarity_threshold = cosine_similarity_threshold
        os.makedirs(f"{self.output_address}", exist_ok=True)        
    
    def query(self, query, embedding_file_name = "documents.json", method="cosine_similarity"):
        selected_chunks = self._retrieve(query, embedding_file_name = embedding_file_name, method = method)
        content = [{"content": chunk["metadata"]["content"], "score": chunk["score"]} for chunk in selected_chunks]
        system_prompt = f"""
        You are an AI assistant. When answering, you should refer to the following information with scores:
        {content}

        """
        response = self._call_llm(system_prompt, query)

        querylog = {
            "query": query,
            "response": response,
            "chunks": selected_chunks
        }

        self.output_file = os.path.join(self.output_address, f"{datetime.now().strftime("%Y%m%d_%H%M%S") }.json")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {self.output_file}")

        return response, selected_chunks

    def query_with_group(self, query, group_file_name, embedding_file_name="documents.json", method="cosine_similarity"):
        group_1_json_name, group_2_json_name, current_date_time = self._get_two_group(group_file_name, embedding_file_name)
        response_1, selected_chunks_1 = self.query(query, embedding_file_name = group_1_json_name, method = method)
        response_2, selected_chunks_2 = self.query(query, embedding_file_name = group_2_json_name, method = method)
        current_folder = os.path.join(self.output_address, current_date_time)
        os.makedirs(current_folder)
        with open(os.path.join(current_folder, "group_1_output.json"), "w", encoding="utf-8") as f:
            querylog = {
            "query": query,
            "response": response_1,
            "chunks": selected_chunks_1
            }
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        with open(os.path.join(current_folder, "group_2_output.json"), "w", encoding="utf-8") as f:
            querylog = {
            "query": query,
            "response": response_2,
            "chunks": selected_chunks_2
            }
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        comparison = self._compare_response(query, response_1, response_2)
        with open(os.path.join(current_folder, "comparison.json"), "w", encoding="utf-8") as f:
            comparisonlog = {
                "query": query,
                "groups": group_file_name,
                "comparison": comparison
            }
            json.dump(comparisonlog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        
        return response_1, selected_chunks_1, response_2, selected_chunks_2, comparison  

    def query_with_glossary(self, query, embedding_file_name="documents.json", method="cosine_similarity"):      
        selected_glossary_chunks = self._retrieve(query, embedding_file_name = "glossary.json", method="cosine_similarity")
        glossary_content = [{"content": chunk["metadata"]["content"], "score": chunk["score"]} for chunk in selected_glossary_chunks]
        selected_chunks = self._retrieve(query, embedding_file_name = embedding_file_name, method = method)
        content = [{"content": chunk["metadata"]["content"], "score": chunk["score"]} for chunk in selected_chunks]
        system_prompt = f"""
        You are an AI assistant. When answering, you should refer to the following information with scores:
        {content}

        Here's some glossary you can refer to: 
        {glossary_content}

        """
        response = self._call_llm(system_prompt, query)

        querylog = {
            "query": query,
            "response": response,
            "chunks": selected_chunks
        }

        self.output_file = os.path.join(self.output_address, f"{datetime.now().strftime("%Y%m%d_%H%M%S") }.json")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {self.output_file}")
        
        return response, selected_chunks, selected_glossary_chunks
    
    def query_with_group_and_glossary(self, query, group_file_name, embedding_file_name="documents.json", method="cosine_similarity"):
        group_1_json_name, group_2_json_name, current_date_time = self._get_two_group(group_file_name, embedding_file_name)
        response_1, selected_chunks_1, selected_glossary_chunks = self.query_with_glossary(query, embedding_file_name = group_1_json_name, method = method)
        response_2, selected_chunks_2, selected_glossary_chunks = self.query_with_glossary(query, embedding_file_name = group_2_json_name, method = method)
        current_folder = os.path.join(self.output_address, current_date_time)
        os.makedirs(current_folder)
        with open(os.path.join(current_folder, "glossary.json"), "w", encoding="utf-8") as f:
            querylog = {
            "query": query,
            "chunks": selected_glossary_chunks
            }
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        with open(os.path.join(current_folder, "group_1_output.json"), "w", encoding="utf-8") as f:
            querylog = {
            "query": query,
            "response": response_1,
            "chunks": selected_chunks_1
            }
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        with open(os.path.join(current_folder, "group_2_output.json"), "w", encoding="utf-8") as f:
            querylog = {
            "query": query,
            "response": response_2,
            "chunks": selected_chunks_2
            }
            json.dump(querylog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        comparison = self._compare_response(query, response_1, response_2)
        with open(os.path.join(current_folder, "comparison.json"), "w", encoding="utf-8") as f:
            comparisonlog = {
                "query": query,
                "groups": group_file_name,
                "comparison": comparison
            }
            json.dump(comparisonlog, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
        
        return response_1, selected_chunks_1, response_2, selected_chunks_2, comparison  

    def _compare_response(self, query, response_1, response_2):
        system_prompt = """
        You are an advanced analytical assistant specialized in comparing textual data.
        Your task is to analyze two given responses to a specific question. 
        Using the provided question as context, identify and explain:
        1. How well each response addresses the question.
        2. Key differences between the responses.
        3. Key similarities between the responses.
        4. Specific examples or phrases that highlight these differences and similarities.
        5. A summarized conclusion highlighting which response is more complete or better aligned with the question.
        Provide your analysis in a clear and structured format.
        """

        # Create the comparison prompt with the question and responses
        prompt = f"""
        Question:
        {query}
        
        Response 1:
        {response_1}
        
        Response 2:
        {response_2}
        """

        return self._call_llm(system_prompt, prompt)

    
    def _get_two_group(self, group_file_name, embedding_file_name="documents.json"):
        embedding_file_address = os.path.join(self.embedding_address, f"{embedding_file_name}")
        with open(embedding_file_address, 'r', encoding='utf-8') as embedding_file:
            embeddings = json.load(embedding_file)
        group_file_address = os.path.join(self.group_address, f"{group_file_name}")
        with open(group_file_address, 'r', encoding='utf-8') as group_file:
            groups = json.load(group_file)
        group_1_documents = groups['group_1']['documents']
        group_2_documents = groups['group_2']['documents']
        group_1_embeddings = []
        group_2_embeddings = []
        for embedding in embeddings:
            if 'metadata' in embedding and 'document_name' in embedding['metadata']:
                document_name = embedding['metadata']['document_name']
                if document_name in group_1_documents:
                    group_1_embeddings.append(embedding)
                elif document_name in group_2_documents:
                    group_2_embeddings.append(embedding)
        current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_folder = os.path.join(self.embedding_address, current_date_time)
        os.makedirs(current_folder, exist_ok=True) 
        group_1_json_name = f"{current_date_time}/group_1.json"
        group_1_json_address = os.path.join(current_folder, "group_1.json")
        with open(group_1_json_address, "w", encoding="utf-8") as f:
            json.dump(group_1_embeddings, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {group_1_json_address}")
        group_2_json_name = f"{current_date_time}/group_2.json"
        group_2_json_address = os.path.join(current_folder, "group_2.json")
        with open(group_2_json_address, "w", encoding="utf-8") as f:
            json.dump(group_2_embeddings, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
            print(f"Data saved at {group_2_json_address}")
        return group_1_json_name, group_2_json_name, current_date_time
    

    def _retrieve(self, query, embedding_file_name = "documents.json", method="cosine_similarity"):
        embedding_file = os.path.join(self.embedding_address, f"{embedding_file_name}")
        with open(embedding_file, "r", encoding="utf-8") as file:
            datas = json.load(file)
            embeddings = [data["embedding"] for data in datas]
        query_embedding = self._get_embedding(query)
        scores = self._score(query_embedding, embeddings, method)
        chunks = [{"score": score, "metadata": data["metadata"]} for score, data in zip(scores, datas)]
        if method == "cosine_similarity":
            score_threshold = self.cosine_similarity_threshold
        else:
            raise NotImplementedError(f"{method} 方法尚未實現。")
        filtered_chunks = [chunk for chunk in chunks if chunk["score"] > score_threshold]
        top_chunks = sorted(filtered_chunks, key=lambda x: x["score"], reverse=True)[:self.select_chunk_number]

        return top_chunks
            

    def _score(self, query_embedding, document_embeddings, method="cosine_similarity"):
        """
        计算查询嵌入与文档嵌入之间的相似性。

        参数:
            query_embedding (numpy.ndarray): 查询的嵌入向量，形状为 (1, n) 或 (n,)。
            document_embeddings (numpy.ndarray): 文档嵌入向量集合，形状为 (m, n)。
            method (str): 相似性计算方法，默认 "cosine_similarity"。

        返回:
            numpy.ndarray: 相似性分数数组，形状为 (m,)。
        """
        query_embedding = np.array(query_embedding)
        document_embeddings = np.array(document_embeddings)
        # 校验输入
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)  # 转换为二维数组
        if document_embeddings.ndim != 2:
            raise ValueError("document_embeddings 必须是二维数组，形状为 (m, n)。")

        # 支持的计算方法
        if method == "cosine_similarity":
            return cosine_similarity(query_embedding, document_embeddings).flatten()
        elif method == "euclidean_distance":
            distances = np.linalg.norm(document_embeddings - query_embedding, axis=1)
            return -distances  # 返回负值以保持与相似性度量一致
        elif method == "dot_product":
            return np.dot(document_embeddings, query_embedding.T).flatten()
        else:
            raise NotImplementedError(f"{method} 方法尚未实现。")


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
    
if __name__ == "__main__":
    rag = Rag()
    rag.query_with_group_and_glossary(" Impacts of anti-Asian racism on Asian Americans; examples of impacted realms include economic, behavioral, psychological, and social areas.", "20241225_211937.json")
