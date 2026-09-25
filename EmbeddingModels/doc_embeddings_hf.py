from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

loader = TextLoader("docs/ml.txt")

docs = loader.load()
#print(docs)

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
)

text = [
    "Machine Learning is a branch of artificial intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every individual task.",
    "Supervised learning is a type of machine learning where the model is trained on labeled data. The model is trained to make predictions or decisions based on the labeled data.",
    "Unsupervised learning is a type of machine learning where the model is trained on unlabeled data. The model is trained to find patterns or relationships in the data.",
    "Semi-supervised learning is a type of machine learning where the model is trained on a combination of labeled and unlabeled data. The model is trained to make predictions or decisions based on the labeled and unlabeled data.",
    "Reinforcement learning is a type of machine learning where the model is trained to make decisions based on the feedback from the environment.",
    "Traditional machine learning is a type of machine learning where the model is trained on traditional data. The model is trained to make predictions or decisions based on the traditional data.",
    "Deep learning is a type of machine learning where the model is trained on deep data. The model is trained to make predictions or decisions based on the deep data.",
    "Natural language processing is a type of machine learning where the model is trained to understand and generate natural language.",
    "Computer vision is a type of machine learning where the model is trained to understand and generate computer vision.",
]


#text = [doc.page_content for doc in docs]
query = "what is Natural language processing and how is it related to machine learning?"

doc_embeddings = embedding.embed_documents(text)    
query_embeddings = embedding.embed_query(query)

print("Length of doc_embeddings: ", len(doc_embeddings))
print("Dimension of doc_embeddings: ", len(doc_embeddings[0]))

# should be 2d list when passing to cosine_similarity
scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(text[index])
print("similarity score:",score)