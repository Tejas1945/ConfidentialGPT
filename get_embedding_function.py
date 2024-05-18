#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
#from chromadb.utils import embedding_functions
#default_ef = embedding_functions.DefaultEmbeddingFunction()
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    #embeddings = GPT4AllEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
"""
from sentence_transformers import SentenceTransformer

def get_embedding_function():
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    return model
"""
