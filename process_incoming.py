import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests 
import numpy as np
import joblib
def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list
    })
    embedding=r.json()['embeddings']
    return embedding
df=joblib.load("embeddings.joblib")
def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate",json={
        "model":"llama3.2",
        "prompt":prompt,
        "stream":False
    })
    return r.json()

incoming_query=input("Ask a question: ")
query_embedding=create_embedding([incoming_query])[0]
# print(query_embedding)
similarities=cosine_similarity(np.vstack(df['embedding'].values),[query_embedding]).flatten()
max_ind =similarities.argsort()[::-1][0:5]
# print(max_ind)
new_df=df.loc[max_ind]
# print(new_df[['chunk_id','number','text']])
# for index,item in new_df.iterrows():
#     print(index,item['number'],item['text'],item['start'],item['end'])

prompt=f'''
I am teaching Software Engineering / OOAD in this course. Here are video chunks containing video number, start time in seconds, and end time in seconds, the text at that time:
{new_df[['number','start','end','text']].to_json(orient="records")}
--------------------------------------------------------
"{incoming_query}"
User asked this question related to video chunks, you have to answer where and how much content is taught in which video (in which video and at what time) and guide the user to go to that particular video. If user asks unrelated question tell that you can answer only questions related to the course.
'''
with open("prompt.txt","w") as f:
    f.write(prompt)
response=inference(prompt)["response"]
print(response)
with open("response.txt","w") as f:
    f.write(response)