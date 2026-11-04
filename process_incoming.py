import os
import requests
import joblib
import numpy as np
from sentence_transformers import CrossEncoder
from rag_strategies import RAGPipeline


def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text}
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def ask_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]


base_dir = os.path.dirname(os.path.abspath(__file__))
joblib_path = os.path.join(base_dir, "embeddings.joblib")
df = joblib.load(joblib_path)

chunks = []
embeddings = []
for lec, group in df.groupby("number", sort=False):
    for i, (_, row) in enumerate(group.iterrows()):
        chunks.append({
            "id": f"{lec}_{i}",
            "lecture": str(lec).zfill(2),
            "text": row["text"],
            "start": row.get("start", 0),
            "end": row.get("end", 0)
        })
        embeddings.append(row["embedding"])

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
pipeline = RAGPipeline(chunks=chunks, embeddings=embeddings, reranker_model=reranker)


def ask_question(question):
    print("\n" + "=" * 60)
    print("USER QUESTION:", question)
    print("=" * 60)

    query_embedding = create_embedding(question)
    retrieved_ids = pipeline.retrieve_rerank(question, query_embedding, top_k=3)

    retrieved_chunks = [pipeline.chunks[pipeline.id_to_index[cid]] for cid in retrieved_ids]

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"Lecture: {chunk['lecture']} | Timestamp: {chunk['start']} - {chunk['end']}\nText: {chunk['text']}\n"
        )
    context = "\n".join(context_parts)

    prompt = f"""You are an AI teaching assistant answering questions about lecture videos.
Answer using only the retrieved lecture content below.

QUESTION:
{question}

RETRIEVED LECTURE CONTENT:
{context}

INSTRUCTIONS:
1. Use only the lecture content provided.
2. If asked WHERE a topic is taught, specify the lecture number and the starting timestamp.
3. Keep the final answer clear and concise.

ANSWER:"""

    answer = ask_llama(prompt)

    print("\nANSWER:")
    print(answer)
    print("=" * 60)
    return answer


if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or 'exit'): ")
        if user_query.lower() == "exit":
            break
        try:
            ask_question(user_query)
        except Exception as e:
            print("Error:", e)
