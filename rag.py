import os
import json
import time
import numpy as np
from retrying import retry
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

class Rag:
    def __init__(self, 
                 document_address,
                 embedding_address = "embeddings",
                 categories_address = "categories",
                 output_address = "outputs",
                 chunk_size = 400,
                 overlap = 150, 
                 use_summary = True, 
                 model_name = llm_model,
                 embedding_model = embedding_model, 
                 api_key = api_key,
                 select_chunk_number = 10,
                 cosine_similarity_threshold = 0.7):
        self.document_address = document_address
        self.embedding_address = embedding_address
        self.categories_address = categories_address
        self.output_address = output_address
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_summary = use_summary
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.select_chunk_number = select_chunk_number
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.embedding_id = 0
        os.makedirs(f"{self.embedding_address}", exist_ok=True)

        
    def documents_embedding(self):
        files_address = f"{self.categories_address}/{self.document_address}"
        txt_files = [f for f in os.listdir(files_address) if f.endswith('.txt') and os.path.isfile(os.path.join(files_address, f))]
        chunks = []
        for txt_file in txt_files:
            file_path = os.path.join(files_address, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                metadatas = self._chunking(content, txt_file)
                for metadata in metadatas:
                    summary = self._get_summary(metadata["content"])
                    print(f"summary: {summary}")
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
            embedding_folder, embedding_file = self._get_embedding_folder_and_file()
            print(f"Creating folder at {embedding_folder}...")
            os.makedirs(f"{embedding_folder}", exist_ok=True)
            print(f"Creating embedding file at...{embedding_file}")
            with open(embedding_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)  # ensure_ascii=False 保留中文，indent=4 格式化
                print(f"Data saved at {embedding_file}")
    
    def query(self, query, method="cosine_similarity"):
        selected_chunks = self._retrieve(query, method)
        content = [{"content": chunk["metadata"]["content"], "score": chunk["score"]} for chunk in selected_chunks]
        system_prompt = f"""
        You are an AI assistant. When answering, you should refer to the following information with scores:
        {content}

        """
        response = self._call_llm(system_prompt, query)

        return response, selected_chunks

    def _retrieve(self, query, method="cosine_similarity"):
        embedding_folder, embedding_file = self._get_embedding_folder_and_file()
        with open(embedding_file, "r", encoding="utf-8") as file:
            datas = json.load(file)
            document_embeddings = [data["embedding"] for data in datas]
        query_embedding = self._get_embedding(query)
        scores = self._score(query_embedding, document_embeddings, method)
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

    def _get_embedding_folder_and_file(self):
        embedding_folder = os.path.join(self.embedding_address, self.document_address)
        embedding_file = os.path.join(embedding_folder, "embedding.json")
        return embedding_folder, embedding_file
                    


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
    rag = Rag("Before 2020-05-01",categories_address = "test_output")
    response, chunk = rag.query("What problem has happened?")
    print(response)



    
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
