import os
import json
import time
import requests
import ssl
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
import numpy as np

# -----------------------------
# Constants
# -----------------------------
OUTPUT_FILE = "../data/embedded_products.json"

EMBED_MODEL = "mxbai-embed-large"
# Change to 'http://' if your server does not support HTTPS
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"

# -----------------------------
# Requests Session with custom SSL context
# -----------------------------
session = requests.Session()
session.trust_env = False

# Create an SSL context that skips hostname verification and disables cert verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Mount the adapter with custom SSL context
adapter = HTTPAdapter()
adapter.init_poolmanager = lambda *args, **kwargs: PoolManager(
    *args,
    ssl_context=ssl_context,
    **kwargs
)
session.mount('http://', adapter)

# -----------------------------
# Generate Embedding
# -----------------------------
def get_embedding(text):
    try:
        response = session.post(
            OLLAMA_URL,
            json={
                "model": EMBED_MODEL,
                "input": text[:5000]
            },
            timeout=20,
            verify=False  # Disable SSL verification for local server
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
    except Exception as e:
        print("\nEmbedding Error:")
        print(e)
        return None


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def getSimilarProducts(text):
    try:
        embedded_products = []
        products = []
        result = []
        top_products = []
        question_embedding = get_embedding(text)
        with open("../data/embedded_products.json", "r", encoding="utf-8") as f:
            embedded_products = json.load(f)
        
        
        for product in embedded_products:
            score = cosine_similarity(question_embedding, product["embedding"])
            new_product = {
                "name": product["name"],
                "price": product["price"],
                "description": product["description"],
                "ingredients": product["ingredients"],
                "score": score
            }
            products.append(new_product)
        
        result = sorted(products, key=lambda x: x["score"], reverse=True)
        top_products = result[:3]
        
        return top_products
    
    except Exception as e:
        print("\nEmbedding Error:")
        print(e)
        return None
            
            