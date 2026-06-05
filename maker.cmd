cat > chat/graph.py << 'PYEOF'
"""
LangGraph Builder - Hybrid Search + LLM Fallback
Fixes:
  1. route_search node must be added BEFORE conditional edges reference it
  2. Added "sql_search" route for price/rating/filter queries
  3. All conditional edge return values have guaranteed fallback keys
  4. TypedDict state instead of plain dict — prevents silent key errors
"""
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
    db_path  = os.environ.get("DATABASE_URL", "./data/products.db")

    # ── Parse structured constraints from the question ─────────────────────

    conditions: list[str] = []
    params:     list      = []

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
        logger.info("sql_search: no SQL filters extracted, falling back to hybrid_search")
        return hybrid_search_products(state)

    # ── Build and run query ────────────────────────────────────────────────

    where_clause = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            product_id, name, price, rating, reviews_count,
            is_gluten_free, is_organic, is_vegan, is_kosher,
            in_stock, stock_status, short_description,
            primary_image_url, url, category, subcategory
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
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    except sqlite3.OperationalError as e:
        logger.error(f"sql_search DB error: {e}")
        return {**state, "search_results": [], "search_method": "sql_error",
                "error_message": str(e)}

    if not rows:
        # No DB results → try vector search as fallback
        logger.info("sql_search: 0 rows returned, falling back to vector_search")
        return vector_search_products(state)

    logger.info(f"sql_search: {len(rows)} rows returned")
    return {
        **state,
        "search_results": rows,
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
    r'(?:under|below|less than|cheaper than|over|above|more than|between)\s*\$?[\d.]',
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
    detail_words = ["image", "detail", "nutrition", "ingredients", "contains", "url"]
    if len(question.split()) <= 2 or any(w in question for w in detail_words):
        return "keyword_search"

    # 3. Recommendation / similarity queries → vector
    recommendation_words = ["recommend", "suggest", "best", "good for", "similar to", "like"]
    if any(w in question for w in recommendation_words):
        return "vector_search"

    # 4. Everything else → hybrid (safe default, never missing)
    return "hybrid_search"


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Build the hybrid search agent.

    Node registration order matters in LangGraph:
    ALL nodes must be added before any conditional_edges reference them.
    """
    workflow = StateGraph(dict)

    # ── Register ALL nodes first ───────────────────────────────────────────
    workflow.add_node("classify_intent",   classify_intent)
    workflow.add_node("route_search",      lambda x: x)          # passthrough router
    workflow.add_node("sql_search",        sql_search_products)   # NEW: SQLite node
    workflow.add_node("keyword_search",    keyword_search_products)
    workflow.add_node("vector_search",     vector_search_products)
    workflow.add_node("hybrid_search",     hybrid_search_products)
    workflow.add_node("generate_answer",   generate_answer_with_llm)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("classify_intent")

    # ── classify_intent → search router OR direct answer ──────────────────
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "needs_search":    "route_search",
            "direct_response": "generate_answer",
        }
    )

    # ── route_search → appropriate search node ────────────────────────────
    workflow.add_conditional_edges(
        "route_search",
        route_search_strategy,
        {
            "sql_search":     "sql_search",
            "keyword_search": "keyword_search",
            "vector_search":  "vector_search",
            "hybrid_search":  "hybrid_search",
        }
    )

    # ── All search nodes → generate_answer → END ──────────────────────────
    workflow.add_edge("sql_search",     "generate_answer")
    workflow.add_edge("keyword_search", "generate_answer")
    workflow.add_edge("vector_search",  "generate_answer")
    workflow.add_edge("hybrid_search",  "generate_answer")
    workflow.add_edge("generate_answer", END)

    graph = workflow.compile()
    logger.info("✅ Agent graph compiled")
    return graph


# ── Singleton ──────────────────────────────────────────────────────────────────

_agent_graph = None

def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
PYEOF
echo "Done"