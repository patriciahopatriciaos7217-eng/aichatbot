import os
import json
import time
import requests
import ssl
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter

# -----------------------------
# Constants
# -----------------------------
INPUT_FILE = "../data/detailInfo.json"
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
# Load Products
# -----------------------------
def load_products():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products")
    return products

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

# -----------------------------
# Build Embeddings
# -----------------------------
def generate_embeddings(products):
    embedded_products = []
    total = len(products)
    for index, product in enumerate(products):
        text = f"""
        Product Name:
        {product.get('name', '')}

        Description:
        {product.get('description', '')}

        Ingredients:
        {product.get('ingredients', '')}

        Price:
        {product.get('price', '')}

        URL:
        {product.get('url', '')}
        """

        embedding = get_embedding(text)
        if embedding is None:
            print(f"[{index+1}/{total}] FAILED")
            continue

        product["embedding"] = embedding
        embedded_products.append(product)
        print(f"[{index+1}/{total}] Embedded -> {product.get('name', 'Unknown')}")
        time.sleep(0.1)
    return embedded_products

# -----------------------------
# Save JSON
# -----------------------------
def save_embedded_products(products):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print("\nFinished")
    print(f"Saved {len(products)} products")
    print(f"File: {OUTPUT_FILE}")

# -----------------------------
# Main
# -----------------------------
def main():
    products = load_products()
    embedded_products = generate_embeddings(products)
    save_embedded_products(embedded_products)

if __name__ == "__main__":
    main()
