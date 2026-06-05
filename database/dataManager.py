import logging
import sqlite3
import json
import chromadb
import ollama
import os
import time
import requests
import ssl
from typing import List, Dict, Optional, Any
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter

# Global connections
SQLITE_CONN = None
CHROMA_COLLECTION = None

# -----------------------------
# CONSTANTS
# -----------------------------
EMBED_MODEL = "nomic-embed-text"  # Changed to more stable model
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
BATCH_SIZE = 5  # Process 5 products at a time
RETRY_COUNT = 3
DELAY_BETWEEN_BATCHES = 0.5

# -----------------------------
# REQUESTS SESSION with custom SSL context
# -----------------------------
session = requests.Session()
session.trust_env = False

# Create an SSL context that skips hostname verification
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


logger = logging.getLogger(__name__)


# ============================================================
# OLLAMA EMBEDDING FUNCTION
# ============================================================


class OllamaEmbeddingFunction:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.dimension = 768

    def _call_ollama(self, text: str) -> List[float]:
        """Call Ollama API for a single text, return embedding or zero vector."""
        try:
            resp = session.post(
                OLLAMA_EMBED_URL,
                json={"model": self.model_name, "input": text[:3000]},
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embeddings", [])
            if emb and isinstance(emb[0], List):
                # Successful batch embedding (we only sent one text but it may be in a list)
                return emb
            elif isinstance(emb, List) and all(isinstance(x, (int, float)) for x in emb):
                # Already a flat embedding
                return emb
            else:
                raise ValueError("Unexpected embedding format")
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            return [0.0] * self.dimension

    def embed_query(self, text: Optional[str] = None, **kwargs) -> List[float]:
        """
        Required by ChromaDB. Accepts 'input' as keyword or positional.
        Always returns a list of floats.
        """
        # ChromaDB may call with 'input' as a list of one string – handle that safely
        raw_input = kwargs.get("input", text)
        if isinstance(raw_input, list):
            # If it's a list, take the first element (should be a string)
            query_text = str(raw_input[0]) if raw_input else ""
        else:
            query_text = str(raw_input) if raw_input else ""

        if not query_text.strip():
            return [0.0] * self.dimension

        return self._call_ollama(query_text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Used by ChromaDB for adding documents – processes in small batches."""
        return self(texts)

    def __call__(self, input):
        """
        ChromaDB calls this with a list of strings when adding documents.
        Processes in batches to avoid overwhelming Ollama.
        """
        if isinstance(input, str):
            input = [input]

        all_embeddings = []
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i: i + BATCH_SIZE]
            try:
                resp = session.post(
                    OLLAMA_EMBED_URL,
                    json={"model": self.model_name,
                          "input": [t[:3000] for t in batch]},
                    timeout=60,
                    verify=False,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                # If the API returns a flat list when batch size =1, wrap it
                if len(batch) == 1 and isinstance(embeddings, list) and all(isinstance(x, (int, float)) for x in embeddings):
                    embeddings = [embeddings]
                # If the number of embeddings doesn't match, fall back to single calls
                if len(embeddings) != len(batch):
                    raise ValueError("Mismatched embedding count")
                all_embeddings.extend(embeddings)
                print(f"✅ Batch {i//BATCH_SIZE + 1} completed")
            except Exception as e:
                print(f"⚠️ Batch failed, doing single calls: {e}")
                for text in batch:
                    all_embeddings.append(self._call_ollama(text))
                    time.sleep(0.1)
            time.sleep(DELAY_BETWEEN_BATCHES)
        return all_embeddings

# ============================================================
# DATABASE CONNECTION
# ============================================================


def get_connection():
    """Get SQLite connection - automatically reconnects if closed"""
    global SQLITE_CONN

    if SQLITE_CONN is None:
        SQLITE_CONN = sqlite3.connect(
            "./king_arthur.db", check_same_thread=False, timeout=10)
        SQLITE_CONN.row_factory = sqlite3.Row
        return SQLITE_CONN

    try:
        SQLITE_CONN.cursor().execute("SELECT 1")
        return SQLITE_CONN
    except Exception:
        print("⚠️ Database connection died, reconnecting...")
        SQLITE_CONN = sqlite3.connect(
            "./king_arthur.db", check_same_thread=False, timeout=10)
        SQLITE_CONN.row_factory = sqlite3.Row
        return SQLITE_CONN


def close_connection():
    """Close database connection - call only when app shuts down"""
    global SQLITE_CONN
    if SQLITE_CONN:
        SQLITE_CONN.close()
        SQLITE_CONN = None
        print("✅ Database connection closed")


# ============================================================
# KEYWORD EXTRACTION & SMART SEARCH
# ============================================================

# Stop words to ignore
STOP_WORDS = {
    'i', 'want', 'to', 'see', 'the', 'a', 'an', 'and', 'or', 'but',
    'for', 'of', 'with', 'without', 'on', 'at', 'by', 'in', 'out',
    'up', 'down', 'from', 'get', 'show', 'tell', 'me', 'my', 'your',
    'please', 'can', 'could', 'would', 'should', 'do', 'does', 'is',
    'are', 'was', 'were', 'have', 'has', 'had', 'be', 'been', 'being',
    'give', 'find', 'search', 'look', 'need', 'like', 'want', 'help'
}

# Product types for prioritization
PRODUCT_TYPES = ['brownie', 'pizza', 'cake', 'bread', 'cookie', 'muffin',
                 'babka', 'scone', 'roll', 'biscuit', 'waffle', 'pancake',
                 'cinnamon', 'donut', 'croissant', 'bagel', 'brioche']

# Dietary keywords for prioritization
DIETARY_KEYWORDS = ['gluten', 'gluten-free', 'gluten free', 'vegan', 'dairy-free',
                    'dairy free', 'nut-free', 'nut free', 'organic', 'kosher',
                    'non-gmo', 'non gmo', 'sugar-free', 'sugar free']


def _normalize(s: str) -> str:
    """Lowercase, replace hyphens/underscores with spaces, collapse whitespace."""
    if not s:
        return ""
    s = s.lower().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _stem(word: str) -> str:
    """Light English plural stemmer. Leaves short words alone."""
    if len(word) <= 3:
        return word
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word

def extract_keywords(question: str) -> List[str]:
    """Extract keywords with hyphen, plural, and multi-word phrase handling."""
    import re
    normalized = _normalize(question)

    detected_dietary = [
        _normalize(dk) for dk in DIETARY_KEYWORDS if _normalize(dk) in normalized
    ]

    detected_types = [
        pt for pt in PRODUCT_TYPES
        if re.search(rf"\b{re.escape(pt)}s?\b", normalized)
    ]

    words = re.findall(r"\b[a-z]{3,}\b", normalized)
    meaningful = [_stem(w) for w in words if w not in STOP_WORDS]

    seen = set()
    ordered = []
    for k in detected_types + detected_dietary + meaningful:
        if k and k not in seen:
            seen.add(k)
            ordered.append(k)

    return ordered if ordered else [normalized]


def search_with_keywords(keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """Flexible OR search across fields; hyphen-insensitive; field-weighted ranking."""
    conn = get_connection()
    cursor = conn.cursor()
    if not keywords:
        return []

    conditions, params = [], []
    for kw in keywords:
        pattern = f"%{kw}%"
        conditions.append(
            "(REPLACE(LOWER(p.name), '-', ' ') LIKE ?"
            " OR REPLACE(LOWER(p.description), '-', ' ') LIKE ?"
            " OR REPLACE(LOWER(p.ingredients), '-', ' ') LIKE ?"
            " OR REPLACE(LOWER(p.details), '-', ' ') LIKE ?)"
        )
        params.extend([pattern, pattern, pattern, pattern])

    where_clause = " OR ".join(conditions)
    query = f'''
        SELECT DISTINCT p.*,
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE {where_clause}
        GROUP BY p.id
        LIMIT ?
    '''
    params.append(limit * 3)
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]

    for r in results:
        r['image_list'] = r['image_list'].split(',') if r.get('image_list') else []

    import re
    for r in results:
        name = _normalize(r.get('name') or '')
        desc = _normalize(r.get('description') or '')
        ing  = _normalize(r.get('ingredients') or '')
        score, matched = 0, 0
        for i, kw in enumerate(keywords):
            pos_w = max(1, len(keywords) - i)
            hit = False
            if kw in name:
                score += 5 * pos_w
                if re.search(rf"\b{re.escape(kw)}\b", name):
                    score += 3
                hit = True
            if kw in desc:
                score += 2 * pos_w
                hit = True
            if kw in ing:
                score += 1 * pos_w
                hit = True
            if hit:
                matched += 1
        r['relevance_score'] = score
        r['match_percentage'] = int((matched / len(keywords)) * 100) if keywords else 0

    results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return results[:limit]


def search_products_smart(search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Smart search with multi-stage fallback (keywords → stemmed words longest-first)."""
    keywords = extract_keywords(search_term)
    print(f"🔍 Extracted keywords: {keywords}")

    results = search_with_keywords(keywords, limit=limit)
    if results:
        return results

    import re
    raw_words = re.findall(r"\b[a-z]{4,}\b", _normalize(search_term))
    candidates = sorted(
        {_stem(w) for w in raw_words if w not in STOP_WORDS},
        key=len,
        reverse=True,
    )
    for word in candidates:
        partial = search_with_keywords([word], limit=limit)
        if partial:
            return partial

    return []



# ============================================================
# INITIALIZATION
# ============================================================

def init_learning_tables():
    """Initialize tables for learning and patterns"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_term TEXT,
            product_name TEXT,
            frequency INTEGER DEFAULT 1,
            last_searched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(search_term, product_name)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            rating INTEGER,
            user_correction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learned_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            correct_answer TEXT,
            confidence INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    print("✅ Learning tables initialized")


def init_sqlite():
    """Initialize SQLite tables - Products and Images are separate but joined"""
    conn = get_connection()
    cursor = conn.cursor()

    # Products table (main product information)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_code TEXT UNIQUE,
            name TEXT NOT NULL,
            price TEXT,
            description TEXT,
            details TEXT,
            ingredients TEXT,
            contains TEXT,
            rating INTEGER,
            nutrition_link TEXT,
            review INTEGER,
            url TEXT,
            image_list ARRAY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
     # Product images table (separate table for images, linked by product_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS product_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            image_url TEXT,
            image_order INTEGER DEFAULT 0,
            FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
        )
    ''')

    # Create index for faster image lookup
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_product_images_product_id 
        ON product_images(product_id)
    ''')

    init_learning_tables()

    conn.commit()
    print("✅ SQLite initialized (Products + Images tables)")


def init_chromadb():
    """Initialize ChromaDB with Ollama embeddings"""
    global CHROMA_COLLECTION

    try:
        client = chromadb.PersistentClient(path="./chroma_db")

        try:
            client.delete_collection("king_arthur_mixes")
        except:
            pass

        CHROMA_COLLECTION = client.create_collection(
            name="king_arthur_mixes",
            embedding_function=OllamaEmbeddingFunction()
        )

        print("✅ ChromaDB initialized")
    except Exception as e:
        print(f"ChromaDB error: {e}")


# ============================================================
# SAMPLE DATA
# ============================================================

def get_sample_data() -> List[Dict[str, Any]]:
    """Return sample product data for testing"""
    products = []
    with open("./data/detailInfo.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    return products


# ============================================================
# PRODUCT STORAGE
# ============================================================


def store_products(products: List[Dict]) -> int:
    """Store products in both databases with BATCHING to prevent Ollama crash"""
    conn = get_connection()
    cursor = conn.cursor()

    chroma_docs = []
    chroma_ids = []
    chroma_metadatas = []
    saved_count = 0

    # First, store all products in SQLite and prepare ChromaDB data
    for idx, product in enumerate(products):
        try:
            product_code = product.get('id', '')
            if product_code.startswith('#'):
                product_code = product_code[1:]

            # Store in SQLite
            cursor.execute('''
                INSERT OR REPLACE INTO products 
                (product_code, name, price, description, details, 
                 ingredients, contains, rating, nutrition_link, review, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_code,
                product.get('name'),
                product.get('price'),
                product.get('description', ''),
                product.get('details', ''),
                product.get('ingredients', ''),
                product.get('contains', ''),
                product.get('rating', ''),
                product.get('nutrition_link', ''),
                product.get('review', ''),
                product.get('url', ''),
            ))

            product_id = cursor.lastrowid

            # Store images
            image_list = product.get('image_list', [])
            for img_idx, image_url in enumerate(image_list):
                cursor.execute('''
                    INSERT OR IGNORE INTO product_images (product_id, image_url, image_order)
                    VALUES (?, ?, ?)
                ''', (product_id, image_url, img_idx))

            # Prepare for ChromaDB
            search_text = f"""
            Product Name: {product.get('name', '')}
            Description: {product.get('description', '')}
            Ingredients: {product.get('ingredients', '')}
            Contains: {product.get('contains', '')}
            """

            chroma_docs.append(search_text)
            chroma_ids.append(f"product_{idx}")
            chroma_metadatas.append({
                "name": product.get('name', ''),
                "price": product.get('price', ''),
                "rating": product.get('rating', ''),
                "description": product.get('description', ''),
                "review": product.get('review', ''),
                "ingredients": product.get('ingredients', ''),
                "url": product.get('url', ''),
                "images": product.get("image_list", []),
                "product_code": product_code,
                "db_id": product_id
            })

            saved_count += 1

            # Progress indicator
            if saved_count % 10 == 0:
                print(f"📦 Prepared {saved_count}/{len(products)} products")

        except Exception as e:
            print(f"⚠️ Error preparing {product.get('name', 'Unknown')}: {e}")

    conn.commit()
    print(f"✅ Stored {saved_count} products in SQLite")

    # ============================================================
    # STORE IN CHROMADB WITH SMALL BATCHES
    # ============================================================

    if chroma_docs and CHROMA_COLLECTION:
        import time

        BATCH_SIZE = 5  # ← SMALL BATCH - prevents Ollama crash!
        total_batches = (len(chroma_docs) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"🔄 Storing {len(chroma_docs)} documents to ChromaDB in {total_batches} batches...")

        for i in range(0, len(chroma_docs), BATCH_SIZE):
            batch_docs = chroma_docs[i:i+BATCH_SIZE]
            batch_ids = chroma_ids[i:i+BATCH_SIZE]
            batch_metadatas = chroma_metadatas[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            try:
                CHROMA_COLLECTION.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                print(
                    f"✅ Batch {batch_num}/{total_batches} completed ({len(batch_docs)} items)")

                # IMPORTANT: Delay between batches to let Ollama recover
                time.sleep(0.5)  # 0.5 second delay

            except Exception as e:
                print(f"⚠️ Batch {batch_num} failed: {e}")

                # Try with even smaller batch
                if len(batch_docs) > 2:
                    print(f"   Retrying with smaller batch...")
                    for j in range(0, len(batch_docs), 2):
                        sub_batch = batch_docs[j:j+2]
                        sub_ids = batch_ids[j:j+2]
                        sub_metas = batch_metadatas[j:j+2]
                        try:
                            CHROMA_COLLECTION.add(
                                documents=sub_batch,
                                ids=sub_ids,
                                metadatas=sub_metas
                            )
                            time.sleep(0.3)
                        except:
                            print(f"   ⚠️ Failed to store individual items")
                else:
                    print(f"   ⚠️ Failed to store batch, skipping...")

    print(f"✅ Completed! Stored {saved_count} products")
    return saved_count

# ============================================================
# LEARNING FUNCTIONS
# ============================================================


def track_search_pattern(search_term: str, product_name: str):
    """Track which products are commonly searched together"""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO search_patterns (search_term, product_name, frequency, last_searched)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(search_term, product_name) 
            DO UPDATE SET 
                frequency = frequency + 1,
                last_searched = CURRENT_TIMESTAMP
        ''', (search_term.lower(), product_name))

        conn.commit()
        print(f"📊 Tracked: '{search_term}' → '{product_name}'")
    except Exception as e:
        print(f"Error tracking pattern: {e}")


def get_related_search_terms(search_term: str, limit: int = 5) -> List[str]:
    """Get related search terms based on past patterns"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT product_name, frequency
        FROM search_patterns 
        WHERE search_term LIKE ? OR product_name LIKE ?
        ORDER BY frequency DESC
        LIMIT ?
    ''', (f'%{search_term}%', f'%{search_term}%', limit))

    results = [row['product_name'] for row in cursor.fetchall()]

    return results


def save_feedback(question: str, answer: str, rating: int, user_correction: str = None):
    """Save user feedback for learning"""
    conn = get_connection()
    cursor = conn.cursor()

    logger.info(
        f"question: {question}, answer: {answer}, rating: {rating}, user_correction:{user_correction}")

    cursor.execute('''
        INSERT INTO feedback (question, answer, rating, user_correction)
        VALUES (?, ?, ?, ?)
    ''', (question, answer, rating, user_correction))

    conn.commit()
    print(f"📝 Feedback saved: Rating {rating}")


def save_learned_response(question: str, correct_answer: str):
    """Save a learned correct response for future use"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO learned_responses (question, correct_answer, confidence)
        VALUES (?, ?, 1)
        ON CONFLICT(question) DO UPDATE SET 
            correct_answer = ?,
            confidence = confidence + 1,
            updated_at = CURRENT_TIMESTAMP
    ''', (question.lower(), correct_answer, correct_answer))

    conn.commit()
    print(f"📚 Learned response for: '{question}'")


def get_learned_response(question: str) -> Optional[str]:
    """Check if we have a learned response for this question"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT correct_answer, confidence
        FROM learned_responses 
        WHERE question = ? AND confidence > 0
        ORDER BY confidence DESC
        LIMIT 1
    ''', (question.lower(),))

    row = cursor.fetchone()

    return row['correct_answer'] if row else None


def get_learning_stats() -> Dict[str, int]:
    """Get statistics about what the agent has learned"""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM search_patterns")
    stats['search_patterns'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1")
    stats['positive_feedback'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 0")
    stats['negative_feedback'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM learned_responses")
    stats['learned_responses'] = cursor.fetchone()[0]

    return stats


# ============================================================
# RETRIEVAL FUNCTIONS (WITH JOINED IMAGES)
# ============================================================

def get_all_products(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all products from SQLite with their images (JOINed)"""
    conn = get_connection()
    cursor = conn.cursor()

    query = '''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
    '''
    if limit:
        query += f' GROUP BY p.id LIMIT {limit}'
    else:
        query += ' GROUP BY p.id'

    cursor.execute(query)

    products = []
    for row in cursor.fetchall():
        product = dict(row)
        # Convert image_list from comma-separated string back to list
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        products.append(product)

    return products


def get_product_count() -> int:
    """Get total number of products"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    return cursor.fetchone()[0]


def search_sqlite(search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search SQLite by name, description, or ingredients (with images)"""
    conn = get_connection()
    cursor = conn.cursor()

    search_pattern = f"%{search_term}%"
    print(search_pattern)
    cursor.execute('''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE p.name LIKE ? 
           OR p.description LIKE ? 
           OR p.ingredients LIKE ?
           OR p.details LIKE ?
        GROUP BY p.id
        LIMIT ?
    ''', (search_pattern, search_pattern, search_pattern, search_pattern, limit))

    products = []
    for row in cursor.fetchall():
        product = dict(row)
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        products.append(product)
    print("products", cursor.fetchall())
    return products


def ensure_string(query):
    """Convert query to string if it's a list"""
    if isinstance(query, list):
        return ' '.join(query) if query else ""
    return str(query) if query else ""


def search_chromadb(query: str, k: int = 5) -> List[Dict]:
    """Search ChromaDB semantically"""
    if not CHROMA_COLLECTION:
        return []

    try:
        query = ensure_string(query)
        results = CHROMA_COLLECTION.query(
            query_texts=[query],
            n_results=k
        )

        if results['documents'] and results['documents'][0]:
            return [
                {
                    'document': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
    except Exception as e:
        print(f"ChromaDB search error: {e}")

    return []


def hybrid_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Combine SQLite and ChromaDB search"""
    sql_results = search_sqlite(query, k)

    if sql_results:
        return sql_results

    if isinstance(query, list):
        query = ' '.join(query)
    chroma_results = search_chromadb(query, k)

    if chroma_results:
        enhanced = []
        for result in chroma_results:
            product_name = result['metadata'].get('name')
            if product_name:
                details = search_sqlite(product_name, 1)
                if details:
                    enhanced.append(details[0])
                else:
                    enhanced.append({'name': product_name})
        return enhanced

    return []


def fast_search(query: str, k: int = 5) -> List[Dict]:
    """Fast search using only SQLite (with images)"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE p.name LIKE ? OR p.description LIKE ? OR p.ingredients LIKE ?
        GROUP BY p.id
        LIMIT ?
    ''', (f'%{query}%', f'%{query}%', f'%{query}%', k))

    results = []
    for row in cursor.fetchall():
        product = dict(row)
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        results.append(product)

    if not results and len(query) > 3:
        words = query.split()[:2]
        for word in words:
            if len(word) > 3:
                cursor.execute('''
                    SELECT p.*, 
                           GROUP_CONCAT(DISTINCT pi.image_url) as image_list
                    FROM products p
                    LEFT JOIN product_images pi ON p.id = pi.product_id
                    WHERE p.name LIKE ?
                    GROUP BY p.id
                    LIMIT ?
                ''', (f'%{word}%', k))
                results = []
                for row in cursor.fetchall():
                    product = dict(row)
                    if product.get('image_list'):
                        product['image_list'] = product['image_list'].split(
                            ',')
                    else:
                        product['image_list'] = []
                    results.append(product)
                if results:
                    break

    return results


def get_product_by_code(product_code: str) -> Optional[Dict[str, Any]]:
    """Get product by product code with its images"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE p.product_code = ?
        GROUP BY p.id
    ''', (product_code,))

    row = cursor.fetchone()

    if row:
        product = dict(row)
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        return product

    return None


def get_product_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get product by exact name with its images"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE p.name = ?
        GROUP BY p.id
    ''', (name,))

    row = cursor.fetchone()

    if row:
        product = dict(row)
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        return product

    return None


def get_all_product_names() -> List[str]:
    """Get list of all product names for context"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM products LIMIT 20")
    return [row['name'] for row in cursor.fetchall()]


def get_product_with_images(product_id: int) -> Optional[Dict[str, Any]]:
    """Get product with all images from the joined tables"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT p.*, 
               GROUP_CONCAT(DISTINCT pi.image_url) as image_list
        FROM products p
        LEFT JOIN product_images pi ON p.id = pi.product_id
        WHERE p.id = ?
        GROUP BY p.id
    ''', (product_id,))

    row = cursor.fetchone()

    if row:
        product = dict(row)
        if product.get('image_list'):
            product['image_list'] = product['image_list'].split(',')
        else:
            product['image_list'] = []
        return product

    return None


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing Database")
    print("=" * 50)

    init_sqlite()
    init_chromadb()

    samples = get_sample_data()
    count = store_products(samples)
    print(f"Stored: {count} products")

    total = get_product_count()
    print(f"Total products: {total}")

    # Test smart search
    print("\n🔍 Testing Smart Search:")
    test_queries = [
        "chocolate",
        "gluten-free brownie",
        "chocolate babka",
        "pizza crust"
    ]

    for query in test_queries:
        results = search_products_smart(query, limit=3)
        print(f"\n  Query: '{query}'")
        for r in results:
            print(
                f"    - {r.get('name')} (Match: {r.get('match_percentage', 0)}%)")
            print(f"      Images: {len(r.get('image_list', []))}")

    # Test learning functions
    print("\n📚 Testing Learning Functions...")
    track_search_pattern("brownie", "Gluten-Free Brownie Mix")
    save_feedback("test question", "test answer", 1)
    stats = get_learning_stats()
    print(f"Learning stats: {stats}")

    print("\n✅ Database test complete!")
