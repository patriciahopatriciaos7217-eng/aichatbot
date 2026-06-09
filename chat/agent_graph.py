"""
LangGraph Builder - LLM-routed search

Flow:
  classify_intent
    → (direct_response) → generate_answer
    → (needs_search)    → analyze_query (LLM picks the strategy)
                            → llm_sql_search   (LLM writes the SQL from schema)
                            → keyword_search
                            → vector_search
                            → hybrid_search
                          → generate_answer → END

The search strategy is chosen by the LLM (analyze_query). When it picks SQL,
llm_sql_search asks the LLM to write a read-only SELECT from the table schema,
validates it (SELECT-only) and runs it on a read-only connection. Every LLM
step falls back to the previous keyword/heuristic behaviour when Ollama is down.
"""
import os
import logging
import re
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from .agent_nodes import (
    classify_intent,
    hybrid_search_products,
    keyword_search_products,
    vector_search_products,
    generate_answer,
)

logger = logging.getLogger(__name__)

_llm = None
_ollama_available = False


def set_llm(llm, ollama_available):
    """Set LLM instance"""
    global _llm, _ollama_available
    _llm = llm
    _ollama_available = ollama_available


def generate_answer_with_llm(state):
    """Wrapper for generate_answer"""
    return generate_answer(state, _llm, _ollama_available)


# ──────────────────────────────────────────────────────────────────────────────
# SQL search node — handles price / rating / stock / dietary filter queries
# ──────────────────────────────────────────────────────────────────────────────

def sql_search_products(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a structured SQLite query built from the parsed question.
    Handles: price filters, rating filters, stock status, dietary flags,
             sorting (cheapest, highest rated), and result limits.
    Falls back to hybrid_search if no SQL filters can be extracted.
    """
    import sqlite3
    import os

    question = state.get("question", "").lower().strip()
    db_path = os.environ.get("DATABASE_URL", "./king_arthur.db")

    # ── Parse structured constraints from the question ─────────────────────

    conditions: list[str] = []
    params:     list = []

    # Price filters  — "under $10", "less than 10", "cheaper than $8", "below $15"
    price_upper = re.search(
        r'(?:under|less than|cheaper than|below|max|maximum|no more than)\s*\$?([\d.]+)',
        question
    )
    price_lower = re.search(
        r'(?:over|more than|above|at least|minimum|min)\s*\$?([\d.]+)',
        question
    )
    price_exact = re.search(r'\$\s*([\d.]+)', question)

    if price_upper:
        conditions.append("price <= ?")
        params.append(float(price_upper.group(1)))
    elif price_lower:
        conditions.append("price >= ?")
        params.append(float(price_lower.group(1)))
    elif price_exact and not price_upper and not price_lower:
        conditions.append("price <= ?")
        params.append(float(price_exact.group(1)))

    # Rating filters — "rating higher than 4", "rated above 4.5", "at least 4 stars"
    rating_match = re.search(
        r'rating\s*(?:higher|more|above|over|greater|at least)?\s*(?:than|of)?\s*([\d.]+)'
        r'|(?:above|over|at least|minimum)\s*([\d.]+)\s*stars?'
        r'|([\d.]+)\s*stars?\s*(?:and above|or higher|or more)?',
        question
    )
    if rating_match:
        val = next(v for v in rating_match.groups() if v is not None)
        conditions.append("rating >= ?")
        params.append(float(val))

    # Stock
    if any(w in question for w in ["in stock", "available", "can buy", "buy now"]):
        conditions.append("in_stock = 1")

    # Dietary flags
    flag_map = {
        "gluten":    "is_gluten_free = 1",
        "organic":   "is_organic = 1",
        "vegan":     "is_vegan = 1",
        "kosher":    "is_kosher = 1",
        "dairy":     "is_dairy_free = 1",
        "non-gmo":   "is_non_gmo = 1",
        "non gmo":   "is_non_gmo = 1",
    }
    for keyword, clause in flag_map.items():
        if keyword in question:
            conditions.append(clause)

    # ── ORDER BY ───────────────────────────────────────────────────────────

    order = "ORDER BY rating DESC"   # sensible default

    if any(w in question for w in ["cheapest", "lowest price", "least expensive", "most affordable"]):
        order = "ORDER BY price ASC"
    elif any(w in question for w in ["most expensive", "highest price", "priciest"]):
        order = "ORDER BY price DESC"
    elif any(w in question for w in ["highest rated", "best rated", "top rated", "most popular"]):
        order = "ORDER BY rating DESC"
    elif any(w in question for w in ["most reviewed", "most reviews"]):
        order = "ORDER BY reviews_count DESC"

    # ── LIMIT ─────────────────────────────────────────────────────────────

    limit = 10   # default
    limit_match = re.search(
        r'(?:show|give|list|get|find|top|first)\s+(\d+)'
        r'|(\d+)\s+(?:products?|items?|results?|mixes?)',
        question
    )
    if limit_match:
        val = next(v for v in limit_match.groups() if v is not None)
        limit = min(int(val), 50)   # cap at 50

    # ── If nothing was extracted → fall back to hybrid search ─────────────
    if not conditions:
        logger.info(
            "sql_search: no SQL filters extracted, falling back to hybrid_search")
        return hybrid_search_products(state)

    # ── Build and run query ────────────────────────────────────────────────

    where_clause = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            id, product_code, name, price, description, details, ingredients, contains, nutrition_link, review, url, created_at, updated_at
        FROM products
        {where_clause}
        {order}
        LIMIT ?
    """
    params.append(limit)

    logger.info(f"sql_search query: {query.strip()} | params: {params}")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. Fetch products
            cursor = conn.execute(query, params)
            products = [dict(row) for row in cursor.fetchall()]
            
            # 2. Extract product IDs
            product_ids = [p['id'] for p in products]
            
            # 3. Fetch images for these product IDs
            if product_ids:
                placeholders = ', '.join('?' * len(product_ids))
                image_query = f"""
                    SELECT product_id, image_url
                    FROM product_images
                    WHERE product_id IN ({placeholders})
                    ORDER BY product_id, image_order
                """
                image_cursor = conn.execute(image_query, product_ids)
                images = image_cursor.fetchall()
                
                # 4. Build image_map: product_id -> list of image URLs
                image_map = {}
                for img in images:
                    pid = img['product_id']
                    if pid not in image_map:
                        image_map[pid] = []
                    image_map[pid].append(img['image_url'])
                
                # 5. Attach image_list to each product
                for p in products:
                    p['image_list'] = image_map.get(p['id'], [])
            else:
                # No products – set empty image_list for each (though loop won't run)
                for p in products:
                    p['image_list'] = []    
    except sqlite3.OperationalError as e:
        logger.error(f"sql_search DB error: {e} → hybrid fallback")
        return hybrid_search_products(state)

    if not products:
        # No DB results → try vector search as fallback
        logger.info(
            "sql_search: 0 products returned, falling back to vector_search")
        return vector_search_products(state)

    logger.info(f"sql_search: {len(products)} products returned")
    return {
        **state,
        "search_results": products,
        "search_method":  "sql",
        "filters_applied": {
            "conditions": conditions,
            "order":      order,
            "limit":      limit,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────

# Keywords that mean "I want structured/filtered results from SQLite"
_SQL_TRIGGERS = [
    # price
    r'\$[\d.]+',
    r'\d+\s*dollars?',
    r'(?:under|below|less than|lower than|more expensive than|cheaper than|over|above|more than|between)\s*\$?[\d.]',
    r'(?:cheapest|most expensive|lowest price|highest price|affordable)',
    # rating
    r'rating\s*(?:higher|above|over|at least|greater)',
    r'(?:above|over|at least)\s*[\d.]+\s*stars?',
    r'highly rated|best rated|top rated',
    # stock
    r'in stock|available now|can(?:\'t)? buy',
    # count
    r'show\s+(?:me\s+)?\d+|top\s+\d+|first\s+\d+|\d+\s+products?',
]
_SQL_PATTERN = re.compile("|".join(_SQL_TRIGGERS), re.IGNORECASE)


def route_after_intent(state: Dict[str, Any]) -> str:
    """
    Route after classify_intent.
    Returns one of: "needs_search" | "direct_response"
    Guaranteed to always return a valid key.
    """
    needs = state.get("needs_search", False)
    # Treat truthy / string "true" / 1 all as needing search
    if needs and str(needs).lower() not in ("false", "0", "no", ""):
        return "needs_search"
    return "direct_response"


def route_search_strategy(state: Dict[str, Any]) -> str:
    """
    Decides which search node to use.
    Returns one of: "sql_search" | "keyword_search" | "vector_search" | "hybrid_search"
    Guaranteed to always return a valid key — no infinite loop possible.
    """
    question = state.get("question", "").lower()

    # 1. Structured filter query → SQLite
    if _SQL_PATTERN.search(question):
        logger.info(f"route → sql_search  (question: {question[:60]})")
        return "sql_search"

    # 2. Short / detail-lookup queries → keyword (fast exact match)
    detail_words = ["detail", "nutrition",
                    "ingredients", "contains", "url"]
    if any(w in question for w in detail_words):
        return "keyword_search"

    # 3. Recommendation / similarity queries → vector
    recommendation_words = ["recommend", "suggest",
                            "best", "good for", "similar to", "like","image"]
    if any(w in question for w in recommendation_words):
        return "vector_search"

    # 4. Everything else → hybrid (safe default, never missing)
    return "hybrid_search"


# ──────────────────────────────────────────────────────────────────────────────
# LLM-driven query analysis + LLM-generated SQL
# ──────────────────────────────────────────────────────────────────────────────

_DB_PATH = os.environ.get("DATABASE_URL", "./king_arthur.db")

# Columns the generated SELECT must always return (so generate_answer/UI work).
_PRODUCT_COLUMNS = (
    "id, product_code, name, price, description, details, ingredients, "
    "contains, rating, nutrition_link, review, url, created_at, updated_at"
)

# Anything that could mutate the DB or chain a second statement → reject.
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM router: choose the search strategy for this question.

    Sets state['search_route'] ∈ {sql, keyword, vector, hybrid}. Falls back to
    the keyword heuristic (route_search_strategy) when Ollama is unavailable or
    the reply can't be parsed.
    """
    question = state.get("question", "")
    steps = state.get("reasoning_steps", []).copy()

    route = None
    if _ollama_available and _llm:
        prompt = (
            "You route questions for a baking-PRODUCTS database assistant.\n"
            "Choose the ONE best search strategy:\n"
            "- sql: structured filters or sorting — price (under $10, cheapest), "
            "rating (4 stars and up), in stock, dietary (gluten-free, vegan, "
            "organic, kosher, dairy-free, non-gmo), counts/limits (top 5), order.\n"
            "- keyword: short exact lookup of a product name or a field "
            "(ingredients, details, nutrition).\n"
            "- vector: recommendation / similarity / vague descriptive intent "
            '("something chocolatey", "similar to X", "good for kids").\n'
            "- hybrid: general product search that doesn't clearly fit the above.\n\n"
            f'Question: "{question}"\n'
            "Reply with ONLY one word: sql, keyword, vector, or hybrid."
        )
        try:
            reply = str(_llm.invoke(prompt)).strip().lower()
            for r in ("sql", "keyword", "vector", "hybrid"):
                if r in reply:
                    route = r
                    break
        except Exception as e:
            logger.error(f"analyze_query LLM error: {e}")

    if route is None:
        route = route_search_strategy(state)   # heuristic node-key fallback
        steps.append(f"🧭 Route (heuristic): {route}")
    else:
        steps.append(f"🧭 Route (LLM): {route}")

    logger.info(f"analyze_query → {route}")
    return {**state, "search_route": route, "reasoning_steps": steps}


def route_from_analysis(state: Dict[str, Any]) -> str:
    """Map the chosen route (LLM word OR heuristic node-key) to a graph node."""
    route = (state.get("search_route") or "hybrid").lower()
    mapping = {
        "sql": "llm_sql_search",          "sql_search": "llm_sql_search",
        "keyword": "keyword_search",      "keyword_search": "keyword_search",
        "vector": "vector_search",        "vector_search": "vector_search",
        "hybrid": "hybrid_search",        "hybrid_search": "hybrid_search",
    }
    return mapping.get(route, "hybrid_search")


def _strip_sql(text: str) -> str:
    """Extract a bare SQL statement from the LLM reply (drop fences/prose)."""
    t = (text or "").strip()
    if "```" in t:
        m = re.search(r"```(?:sql)?\s*(.+?)```", t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    idx = t.lower().find("select")
    if idx > 0:
        t = t[idx:]
    return t.strip().rstrip(";").strip()


def _is_safe_select(sql: str) -> bool:
    """Allow ONLY a single read-only SELECT. Belt to the read-only-connection braces."""
    if not sql:
        return False
    low = sql.lower().strip()
    if not low.startswith("select"):
        return False
    if ";" in sql or "--" in sql or "/*" in sql:   # no statement chaining / comments
        return False
    if _FORBIDDEN_SQL.search(sql):
        return False
    return True


def _generate_sql(question: str) -> Optional[str]:
    """Ask the LLM to write a read-only SELECT from the products schema."""
    if not (_ollama_available and _llm):
        return None
    prompt = (
        "You are a SQLite expert. Write ONE read-only SQL query that answers the "
        "user's question about baking products.\n\n"
        "Table `products` columns:\n"
        "  id INTEGER, product_code TEXT, name TEXT,\n"
        "  price TEXT  -- e.g. '$8.95', a string with a leading $,\n"
        "  description TEXT, details TEXT, ingredients TEXT, contains TEXT,\n"
        "  rating INTEGER  -- 0..5,\n"
        "  nutrition_link TEXT, review TEXT, url TEXT, created_at, updated_at\n\n"
        "Rules:\n"
        "- Output ONLY the SELECT statement. No prose, no markdown, no semicolon.\n"
        "- Query ONLY the products table; it must be read-only.\n"
        f"- Always select exactly these columns: {_PRODUCT_COLUMNS}.\n"
        "- price is TEXT with a '$'; for numeric price filters/sorting use "
        "CAST(REPLACE(price, '$', '') AS REAL).\n"
        "- Dietary flags (gluten-free, vegan, organic, kosher, dairy-free, "
        "non-gmo) are NOT columns — match them with LIKE on name, description, "
        "details and ingredients.\n"
        "- Add a LIMIT (<= 50) when the question doesn't specify a count.\n\n"
        f'Question: "{question}"\n'
        "SQL:"
    )
    try:
        return _strip_sql(str(_llm.invoke(prompt)))
    except Exception as e:
        logger.error(f"_generate_sql LLM error: {e}")
        return None


def _run_select(sql: str) -> List[Dict[str, Any]]:
    """Execute a validated SELECT on a READ-ONLY connection and attach images."""
    import sqlite3
    uri = f"file:{_DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        products = [dict(r) for r in conn.execute(sql).fetchall()]

        ids = [p["id"] for p in products if p.get("id") is not None]
        image_map: Dict[Any, list] = {}
        if ids:
            placeholders = ", ".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT product_id, image_url FROM product_images "
                f"WHERE product_id IN ({placeholders}) "
                f"ORDER BY product_id, image_order",
                ids,
            ).fetchall()
            for row in rows:
                image_map.setdefault(row["product_id"], []).append(row["image_url"])
        for p in products:
            p["image_list"] = image_map.get(p.get("id"), [])
    return products


def llm_sql_search_products(state: Dict[str, Any]) -> Dict[str, Any]:
    """SQL search whose query is written by the LLM from the table schema.

    generate → validate (SELECT-only) → execute (read-only connection) → attach
    images. Falls back to the heuristic sql_search (then hybrid/vector) if
    generation/validation/execution fails or returns nothing.
    """
    question = state.get("question", "")
    steps = state.get("reasoning_steps", []).copy()

    sql = _generate_sql(question)
    if not _is_safe_select(sql or ""):
        steps.append("⚠️ LLM SQL unavailable/unsafe → heuristic SQL search")
        logger.warning(f"Unsafe/empty LLM SQL, falling back. Got: {sql!r}")
        return sql_search_products({**state, "reasoning_steps": steps})

    logger.info(f"llm_sql_search SQL: {sql}")
    try:
        products = _run_select(sql)
    except Exception as e:
        steps.append(f"⚠️ LLM SQL failed ({e}) → heuristic SQL search")
        logger.error(f"llm_sql_search execution error: {e}")
        return sql_search_products({**state, "reasoning_steps": steps})

    if not products:
        steps.append("ℹ️ LLM SQL returned 0 rows → vector search")
        return vector_search_products({**state, "reasoning_steps": steps})

    steps.append(f"✅ LLM SQL search found {len(products)} products")
    return {
        **state,
        "search_results": products,
        "product_count": len(products),
        "suggested_products": [p.get("name") for p in products[:3] if p.get("name")],
        "search_method": "llm_sql",
        "generated_sql": sql,
        "reasoning_steps": steps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Build the LLM-routed search agent.

    Node registration order matters in LangGraph:
    ALL nodes must be added before any conditional_edges reference them.
    """
    workflow = StateGraph(dict)

    # ── Register ALL nodes first ───────────────────────────────────────────
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("analyze_query",   analyze_query)            # LLM router
    workflow.add_node("llm_sql_search",  llm_sql_search_products)  # LLM-written SQL
    workflow.add_node("keyword_search",  keyword_search_products)
    workflow.add_node("vector_search",   vector_search_products)
    workflow.add_node("hybrid_search",   hybrid_search_products)
    workflow.add_node("generate_answer", generate_answer_with_llm)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("classify_intent")

    # ── classify_intent → LLM query analysis OR direct answer ─────────────
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "needs_search":    "analyze_query",
            "direct_response": "generate_answer",
        }
    )

    # ── analyze_query (LLM) → the chosen search node ──────────────────────
    workflow.add_conditional_edges(
        "analyze_query",
        route_from_analysis,
        {
            "llm_sql_search": "llm_sql_search",
            "keyword_search": "keyword_search",
            "vector_search":  "vector_search",
            "hybrid_search":  "hybrid_search",
        }
    )

    # ── All search nodes → generate_answer → END ──────────────────────────
    workflow.add_edge("llm_sql_search", "generate_answer")
    workflow.add_edge("keyword_search", "generate_answer")
    workflow.add_edge("vector_search",  "generate_answer")
    workflow.add_edge("hybrid_search",  "generate_answer")
    workflow.add_edge("generate_answer", END)

    graph = workflow.compile()
    logger.info("✅ Agent graph compiled (LLM-routed)")
    return graph


# ── Singleton ──────────────────────────────────────────────────────────────────

_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
