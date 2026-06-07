"""
LangGraph Nodes with Smart Search and LLM Fallback
Compatible with graph.py — state keys unified:
  search_results  (was: found_products / search_results)
  answer          (unchanged)
  needs_search    (unchanged)
"""
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONVERSATION CONTEXT MEMORY
# ============================================================

CONVERSATION_CONTEXT = {}


def get_conversation_context(session_id: str) -> Dict:
    if session_id not in CONVERSATION_CONTEXT:
        CONVERSATION_CONTEXT[session_id] = {
            "last_search_results": [],
            "last_question": "",
            "last_answer": "",
            "conversation_history": []
        }
    return CONVERSATION_CONTEXT[session_id]


def update_conversation_context(session_id: str, question: str, answer: str, products: List):
    context = get_conversation_context(session_id)
    context["last_question"] = question
    context["last_answer"] = answer
    if products:
        context["last_search_results"] = products
    context["conversation_history"].append({
        "question": question,
        "answer": answer,
        "products": products
    })
    if len(context["conversation_history"]) > 10:
        context["conversation_history"].pop(0)


# ============================================================
# HELPER
# ============================================================

def extract_simple_keywords(question: str) -> str:
    """Extract keywords from question for search"""
    common_words = [
        'i', 'want', 'to', 'see', 'the', 'a', 'an', 'and', 'or',
        'for', 'of', 'with', 'get', 'show', 'tell', 'me', 'please',
        'what', 'is', 'are', 'can', 'you', 'do', 'does', 'have',
        'about', 'details', 'more', 'info', 'information', "show", "image"
    ]
    words = question.lower().split()
    keywords = [w for w in words if w not in common_words and len(w) > 2]
    # Return the FILTERED keywords (not the raw sentence) so callers don't
    # end up doing a single LIKE '%whole question%' match that never hits.
    return ' '.join(keywords) if keywords else question


def _product_from_chroma_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build a product dict from ChromaDB metadata, decoding the images field.

    Images are stored in metadata as a JSON-encoded string (ChromaDB rejects
    list-valued metadata). Handle both the new JSON form and any legacy list.
    """
    import json
    images = meta.get('images', [])
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except Exception:
            images = [images] if images else []
    if not isinstance(images, list):
        images = []
    return {
        'name': meta.get('name', ''),
        'price': meta.get('price', ''),
        'description': meta.get('description', ''),
        'ingredients': meta.get('ingredients', ''),
        'rating': meta.get('rating', ''),
        'url': meta.get('url', ''),
        'product_code': meta.get('product_code', ''),
        'image_list': images,
    }


def _enrich_chroma_results(results: List[Dict]) -> List[Dict[str, Any]]:
    """Turn ChromaDB hits into full product dicts, guaranteeing image_list.

    Prefers the richer SQLite row (joined product_images) and falls back to the
    embedded ChromaDB metadata so a hit is never dropped and images are never
    lost — this is why "vector search returned no images" used to happen.
    """
    from database.dataManager import search_sqlite
    enhanced: List[Dict[str, Any]] = []
    for result in results:
        meta = result.get('metadata', {}) or {}
        name = meta.get('name')
        product = None
        if name:
            details = search_sqlite(name, 1)
            if details:
                product = details[0]
        if product is None:
            product = _product_from_chroma_meta(meta)
        # Guarantee images: if SQLite row had none, recover them from metadata.
        if not product.get('image_list'):
            product['image_list'] = _product_from_chroma_meta(meta).get('image_list', [])
        enhanced.append(product)
    return enhanced


# ============================================================
# RELEVANCE GATE + FLEXIBLE (LLM) ANSWERS + LEARNING
# ============================================================

# Baking vocabulary for the no-LLM relevance heuristic.
_BAKING_TERMS = {
    'bake', 'baking', 'baker', 'mix', 'mixes', 'flour', 'dough', 'batter',
    'recipe', 'recipes', 'cake', 'cookie', 'cookies', 'brownie', 'brownies',
    'bread', 'pizza', 'muffin', 'muffins', 'scone', 'biscuit', 'pancake',
    'waffle', 'babka', 'roll', 'rolls', 'bagel', 'brioche', 'croissant',
    'donut', 'doughnut', 'pastry', 'pie', 'crust', 'frosting', 'icing',
    'glaze', 'yeast', 'sourdough', 'gluten', 'vegan', 'organic', 'kosher',
    'dairy', 'ingredient', 'ingredients', 'allergen', 'nutrition', 'oven',
    'sugar', 'butter', 'egg', 'eggs', 'chocolate', 'vanilla', 'cinnamon',
    'sweet', 'dessert', 'cook', 'cooking', 'food', 'kitchen', 'king arthur',
    'product', 'products', 'price', 'stock', 'rating',
}

# Word-boundary matcher (longest terms first). \b at the start matches plurals
# ("cake" → "cakes") without false positives mid-word ("egg" in "beggar").
_BAKING_PATTERN = re.compile(
    r'\b(' + '|'.join(sorted((re.escape(t) for t in _BAKING_TERMS),
                             key=len, reverse=True)) + r')',
    re.IGNORECASE,
)


def _is_baking_related(question: str, llm=None, ollama_available: bool = False) -> bool:
    """Decide whether a question is about baking / mixes / food.

    Uses the LLM when available (most accurate), otherwise a keyword heuristic.
    Fails OPEN (returns True on uncertainty) so a genuine baking question is
    never wrongly refused.
    """
    q = (question or "").lower().strip()
    if not q:
        return True

    # Clear baking vocabulary → on-topic (also avoids a needless LLM call).
    if _BAKING_PATTERN.search(q):
        return True

    if ollama_available and llm:
        try:
            verdict = llm.invoke(
                "You are a classifier for a baking-mix shopping assistant.\n"
                f'User message: "{question}"\n\n'
                "Is this message about baking, cooking, food, recipes, ingredients, "
                "or baking mixes/products? Answer with ONLY one word: YES or NO."
            )
            return "yes" in str(verdict).strip().lower()[:5]
        except Exception as e:
            logger.error(f"relevance LLM error: {e}")
            return True  # fail open

    # No LLM and no baking keyword → treat as off-topic.
    return False


def _shares_keyword(question: str, products: List[Dict]) -> bool:
    """True if the question shares a specific word with any returned product NAME.

    Safety net so a genuine product query isn't discarded as 'off-topic' (matters
    when the LLM is down). NAME only — matching descriptions causes false
    positives because product prose contains generic words like "company",
    "make", "love". Generic catalog words are also excluded.
    """
    stop = {'the', 'and', 'you', 'have', 'what', 'for', 'with', 'can', 'are',
            'does', 'show', 'tell', 'about', 'this', 'that', 'your', 'from',
            'get', 'any', 'some', 'want', 'need', 'make', 'how', 'where',
            'mix', 'mixes', 'king', 'arthur', 'baking', 'company', 'product',
            'products', 'love', 'good', 'best', 'new', 'please'}
    q_tokens = {w for w in re.findall(r'[a-z]{3,}', (question or '').lower())} - stop
    if not q_tokens:
        return False
    for p in products[:5]:
        name = (p.get('name', '') or '').lower()
        if any(t in name for t in q_tokens):
            return True
    return False


def _flexible_answer(question: str, on_topic: bool, llm=None,
                     ollama_available: bool = False) -> str:
    """Produce a helpful answer when no relevant product was found.

    on_topic=True  → answer the baking question helpfully.
    on_topic=False → politely steer back to baking WITHOUT recommending products.
    Never dumps the product catalog.
    """
    if ollama_available and llm:
        if on_topic:
            prompt = (
                "You are the King Arthur Baking Assistant, a friendly baking expert.\n"
                f'The user asked: "{question}"\n\n'
                "Give a genuinely helpful, concise answer about baking. Do NOT invent "
                "product names, prices, or a catalog list. You may mention that we "
                "carry King Arthur baking mixes, but keep the focus on answering."
            )
        else:
            prompt = (
                "You are the King Arthur Baking Assistant. You ONLY help with baking, "
                "mixes, ingredients, and recipes.\n"
                f'The user said: "{question}"\n\n'
                "This is NOT related to baking. Politely and briefly say you specialize "
                "in baking and King Arthur mixes, and invite a baking-related question. "
                "Do NOT recommend or list any products."
            )
        try:
            return str(llm.invoke(prompt)).strip()
        except Exception as e:
            logger.error(f"flexible answer LLM error: {e}")

    # Static fallback (no LLM) — still no catalog dump.
    if on_topic:
        return (
            "I'm the King Arthur Baking Assistant. I couldn't find a specific product "
            "for that, but I'm happy to help with baking questions, ingredients, and "
            "recipes. Could you tell me a bit more about what you're baking?"
        )
    return (
        "I'm the King Arthur Baking Assistant, so I specialize in baking, mixes, "
        "ingredients, and recipes. I can't help with that one — but ask me anything "
        "about baking, e.g. \"What gluten-free brownie mixes do you have?\""
    )


_REFERENCE_WORDS = {
    'first', 'second', 'third', 'fourth', 'fifth', 'last',
    '1st', '2nd', '3rd', '4th', '5th',
    'detail', 'details', 'more', 'it', 'this', 'that', 'elaborate',
}


def is_cacheable_question(question: str) -> bool:
    """Whether a question is safe to store/serve as a learned response.

    Excludes short, context-dependent queries (bare numbers, ordinals,
    'details', follow-ups) that only make sense relative to previous results —
    caching those would, e.g., make typing "2" always return a stale answer
    instead of selecting product #2.
    """
    q = (question or "").lower().strip()
    if not q or q.isdigit():
        return False
    words = q.split()
    if len(words) < 3:
        return False
    if any(w in _REFERENCE_WORDS for w in words):
        return False
    return True


def _remember_learned(question: str, answer: str) -> None:
    """Persist an informational answer so the bot serves it consistently later."""
    if not is_cacheable_question(question):
        return
    try:
        from database.dataManager import save_learned_response
        save_learned_response(question, answer)
    except Exception as e:
        logger.error(f"save_learned_response failed: {e}")


def _track_patterns(question: str, products: List[Dict]) -> None:
    """Record which products a query surfaced (search-pattern learning)."""
    try:
        from database.dataManager import track_search_pattern
        for p in products[:3]:
            name = p.get('name')
            if name:
                track_search_pattern(question, name)
    except Exception as e:
        logger.error(f"track_search_pattern failed: {e}")


# ============================================================
# INTENT CLASSIFICATION
# FIX: removed session_id parameter — graph calls this as a node
#      with only (state). session_id falls back to "default".
# ============================================================

def classify_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify intent with context awareness.
    Called by LangGraph as a node — must accept only (state).
    """
    session_id = state.get("session_id", "default")   # ← read from state, not arg
    question = state.get('question', '').lower().strip()
    steps = state.get('reasoning_steps', []).copy()

    context = get_conversation_context(session_id)
    last_products = context.get("last_search_results", [])

    # ── Reference to previous products ────────────────────────────────────
    reference_patterns = {
        'first': 0, '1st': 0,
        'second': 1, '2nd': 1,
        'third': 2, '3rd': 2,
        'fourth': 3, '4th': 3,
        'fifth': 4, '5th': 4,
        'last': -1
    }
    for pattern, idx in reference_patterns.items():
        if pattern in question and last_products:
            product_index = len(last_products) - 1 if idx == -1 else idx
            if product_index < len(last_products):
                product = last_products[product_index]
                steps.append(f"🎯 User wants details of {pattern} product: {product.get('name','Unknown')}")
                return {
                    **state,
                    'intent': 'show_product_details',
                    'target_product': product,
                    'needs_search': False,
                    'reasoning_steps': steps,
                }

    # ── Pattern: "tell me about X" ─────────────────────────────────────────
    product_mention_patterns = [
        r'tell me about (.+)', r'about (.+)', r'details of (.+)',
        r'information on (.+)', r'what is (.+)', r'show me (.+)',
        r'can you tell me about (.+)', r'give me details of (.+)'
    ]
    for pattern in product_mention_patterns:
        match = re.search(pattern, question)
        if match:
            product_name = match.group(1).strip()
            for product in last_products:
                if product_name.lower() in product.get('name', '').lower():
                    steps.append(f"🎯 User wants details of: {product.get('name','Unknown')}")
                    return {
                        **state,
                        'intent': 'show_product_details',
                        'target_product': product,
                        'needs_search': False,
                        'reasoning_steps': steps,
                    }

    # ── "details / more info" ──────────────────────────────────────────────
    detail_words = ['details', 'more info', 'tell me more', 'more about', 'elaborate']
    if any(w in question for w in detail_words) and last_products:
        product_list = "\n".join([
            f"{i+1}. {p.get('name','Unknown')}" for i, p in enumerate(last_products)
        ])
        steps.append("❓ Asking which product for details")
        return {
            **state,
            'intent': 'ask_which_product',
            'needs_search': False,
            'direct_response': (
                f"Which product would you like more details about?\n\n"
                f"{product_list}\n\nJust type the number or product name."
            ),
            'reasoning_steps': steps,
        }

    # ── Number reference (1, 2, 3) ─────────────────────────────────────────
    number_match = re.match(r'^(\d+)$', question)
    if number_match and last_products:
        idx = int(number_match.group(1)) - 1
        if 0 <= idx < len(last_products):
            product = last_products[idx]
            steps.append(f"🎯 User wants details of product #{idx+1}: {product.get('name','Unknown')}")
            return {
                **state,
                'intent': 'show_product_details',
                'target_product': product,
                'needs_search': False,
                'reasoning_steps': steps,
            }

    # ── Greeting ───────────────────────────────────────────────────────────
    if question in ['hi', 'hello', 'hey', 'sup', 'howdy']:
        steps.append("👋 Greeting detected")
        return {
            **state,
            'intent': 'greeting',
            'needs_search': False,
            'direct_response': (
                "Hello! 👋 Welcome to the King Arthur Baking Assistant!\n\n"
                "I'm here to help you find baking mixes, check ingredients, and provide instructions.\n\n"
                "**Try asking me:**\n"
                "• \"What gluten-free brownie mixes do you have?\"\n"
                "• \"Show me cake mixes\"\n\n"
                "What can I help you with today? 🍞"
            ),
            'reasoning_steps': steps,
        }

    greeting_phrases = ['good morning', 'good afternoon', 'good evening', 'greetings', 'nice to meet you']
    if any(p in question for p in greeting_phrases):
        steps.append("👋 Greeting phrase detected")
        return {
            **state,
            'intent': 'greeting',
            'needs_search': False,
            'direct_response': "Good day! 👋 Welcome to the King Arthur Baking Assistant. How can I help you with your baking needs today?",
            'reasoning_steps': steps,
        }

    # ── Farewell ───────────────────────────────────────────────────────────
    farewells = ['bye', 'goodbye', 'see you', 'exit', 'quit', 'see ya', 'nice talking to you']
    if question in farewells or any(f in question for f in farewells):
        steps.append("👋 Farewell detected")
        return {
            **state,
            'intent': 'farewell',
            'needs_search': False,
            'direct_response': "Goodbye! Happy baking! 🍰 Come back anytime you have questions about King Arthur Baking products.",
            'reasoning_steps': steps,
        }

    # ── Thanks ─────────────────────────────────────────────────────────────
    thanks_phrases = ['thank you', 'thanks', 'thank', 'appreciate it', 'thx']
    if any(p in question for p in thanks_phrases):
        steps.append("🙏 Thanks detected")
        return {
            **state,
            'intent': 'thanks',
            'needs_search': False,
            'direct_response': "You're very welcome! 😊 Is there anything else baking-related I can assist you with?",
            'reasoning_steps': steps,
        }

    # ── About assistant ────────────────────────────────────────────────────
    about_phrases = ['who are you', 'what are you', 'tell me about yourself',
                     'what can you do', 'your purpose', 'about you', 'introduce yourself']
    if any(p in question for p in about_phrases):
        steps.append("ℹ️ About assistant question")
        return {
            **state,
            'intent': 'about_assistant',
            'needs_search': False,
            'direct_response': (
                "🍞 **About the King Arthur Baking Assistant**\n\n"
                "I'm your AI-powered guide to King Arthur Baking Company's product mixes!\n\n"
                "**What I can help you with:**\n"
                "• 🔍 **Find products** - Search for brownie, pizza, cake, and bread mixes\n"
                "• 🥗 **Dietary needs** - Find gluten-free, vegan, and organic options\n"
                "• 📝 **Ingredients** - Check what's in each mix\n"
                "• 👩‍🍳 **Instructions** - Get baking directions and tips\n"
                "• 💰 **Pricing** - Compare product prices\n\n"
                "What would you like to know about our baking products today?"
            ),
            'reasoning_steps': steps,
        }

    # ── Help ───────────────────────────────────────────────────────────────
    help_phrases = ['help', 'what can you help me with', 'how to use', 'capabilities', 'features']
    if any(p in question for p in help_phrases):
        steps.append("❓ Help requested")
        return {
            **state,
            'intent': 'help',
            'needs_search': False,
            'direct_response': (
                "📚 **How I Can Help You**\n\n"
                "**Product Discovery**\n"
                "• \"What brownie mixes do you have?\"\n"
                "• \"Show me gluten-free options\"\n\n"
                "**Ingredient Information**\n"
                "• \"Does the pizza crust contain dairy?\"\n\n"
                "**Price & Availability**\n"
                "• \"Show me products under $10\"\n"
                "• \"What's the cheapest mix?\"\n\n"
                "**Just type your question naturally!** 🍞"
            ),
            'reasoning_steps': steps,
        }

    # ── Image request ──────────────────────────────────────────────────────
    image_phrases = ['show me image', 'show me picture', 'image of', 'picture of',
                     'what does it look like', 'photo of', 'see the product', 'show image']
    if any(p in question for p in image_phrases):
        steps.append("🖼️ Image request detected")
        return {
            **state,
            'intent': 'image_request',
            'needs_search': True,
            'direct_response': None,
            'reasoning_steps': steps,
        }

    # ── Default: product search ────────────────────────────────────────────
    steps.append("🔍 Product search — will query database")
    return {
        **state,
        'intent': 'product_search',
        'needs_search': True,
        'direct_response': None,
        'reasoning_steps': steps,
    }


# ============================================================
# SEARCH NODES
# FIX: all three nodes now write to state['search_results']
#      (was 'found_products' in original — mismatched with graph)
# ============================================================

def keyword_search_products(state: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword search using SQLite with per-keyword OR matching + ranking.

    Uses search_products_smart (extracts keywords, OR-matches each across
    name/description/ingredients/details, ranks by relevance) instead of a
    single LIKE on the whole sentence — the old approach rarely matched.
    """
    from database.dataManager import search_products_smart, get_all_product_names

    question = state.get('question', '')
    steps = state.get('reasoning_steps', []).copy()
    logger.info(f"🔍 Keyword searching for: {question}")

    results = search_products_smart(question, limit=5)

    if results:
        steps.append(f"✅ Keyword search found {len(results)} products")
        return {
            **state,
            'search_results': results,          # ← unified key
            'product_count': len(results),
            'suggested_products': [p['name'] for p in results[:2]],
            'search_method': 'keyword',
            'reasoning_steps': steps,
        }

    steps.append("⚠️ No keyword matches found")
    product_names = get_all_product_names()
    return {
        **state,
        'search_results': [],
        'product_count': 0,
        'available_products_context': product_names,
        'search_method': 'keyword',
        'reasoning_steps': steps,
    }


def vector_search_products(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pure vector search using ChromaDB (semantic, meaning-based)"""
    from database.dataManager import search_chromadb, get_all_product_names

    question = state.get('question', '')
    steps = state.get('reasoning_steps', []).copy()
    logger.info(f"🔍 Vector searching for: {question}")

    results = search_chromadb(question, k=5)

    if results:
        enhanced = _enrich_chroma_results(results)

        steps.append(f"✅ Vector search found {len(enhanced)} semantically similar products")
        return {
            **state,
            'search_results': enhanced,          # ← unified key
            'product_count': len(enhanced),
            'suggested_products': [p['name'] for p in enhanced[:3]],
            'search_method': 'vector',
            'reasoning_steps': steps,
        }

    steps.append("⚠️ No vector matches found")
    product_names = get_all_product_names()
    return {
        **state,
        'search_results': [],
        'product_count': 0,
        'available_products_context': product_names,
        'search_method': 'vector',
        'reasoning_steps': steps,
    }


def hybrid_search_products(state: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid search: vector first, keyword fallback"""
    from database.dataManager import search_chromadb, search_sqlite, get_all_product_names

    question = state.get('question', '')
    steps = state.get('reasoning_steps', []).copy()
    logger.info(f"🔍 Hybrid searching for: {question}")

    # Step 1: vector
    vector_results = search_chromadb(question, k=5)
    if vector_results:
        enhanced = _enrich_chroma_results(vector_results)

        steps.append(f"✅ Hybrid (vector) found {len(enhanced)} products")
        return {
            **state,
            'search_results': enhanced,          # ← unified key
            'product_count': len(enhanced),
            'suggested_products': [p['name'] for p in enhanced[:3]],
            'search_method': 'hybrid_vector',
            'reasoning_steps': steps,
        }

    # Step 2: keyword fallback (per-keyword OR matching + ranking)
    from database.dataManager import search_products_smart
    keyword_results = search_products_smart(question, limit=5)
    if keyword_results:
        steps.append(f"✅ Hybrid (keyword) found {len(keyword_results)} products")
        return {
            **state,
            'search_results': keyword_results,   # ← unified key
            'product_count': len(keyword_results),
            'suggested_products': [p['name'] for p in keyword_results[:3]],
            'search_method': 'hybrid_keyword',
            'reasoning_steps': steps,
        }

    steps.append("⚠️ No results from hybrid search")
    product_names = get_all_product_names()
    return {
        **state,
        'search_results': [],
        'product_count': 0,
        'available_products_context': product_names,
        'search_method': 'hybrid',
        'reasoning_steps': steps,
    }


# ============================================================
# GENERATE ANSWER
# FIX: removed session_id parameter — graph calls with (state, llm, ollama_available)
#      session_id read from state instead.
#      Reads 'search_results' (unified key) not 'found_products'.
# ============================================================

def generate_answer(state: Dict[str, Any], llm=None, ollama_available: bool = False) -> Dict[str, Any]:
    """
    Generate answer with context awareness.
    Reads search_results (unified key matching all search nodes + graph's sql_search).
    """
    session_id = state.get("session_id", "default")  # ← read from state
    question   = state.get('question', '')
    steps      = state.get('reasoning_steps', []).copy()

    context      = get_conversation_context(session_id)
    last_products = context.get("last_search_results", [])

    # ── Show product details ───────────────────────────────────────────────
    if state.get('intent') == 'show_product_details':
        product = state.get('target_product')
        if product:
            name         = product.get('name', 'Unknown')
            price        = product.get('price', 'N/A')
            description  = product.get('description', 'No description')
            ingredients  = product.get('ingredients', 'No ingredients listed')
            contains     = product.get('contains', 'No allergen info')
            instructions = product.get('instructions', 'No instructions available')
            image_list   = product.get('image_list', [])
            nutrition_link = product.get('nutrition_link', '')

            answer = (
                f"📖 **{name}** - {price}\n\n"
                f"**📝 Description:** {description}\n\n"
                f"**🥗 Ingredients:** {ingredients}\n\n"
                f"**⚠️ Contains:** {contains}\n\n"
                f"**👩‍🍳 Instructions:** {instructions}\n\n"
            )
            for img in image_list[:2]:
                answer += f'<img src="{img}" width="300" style="border-radius:10px;margin:10px 0;">\n\n'
            if nutrition_link:
                answer += f"🔗 [View Nutrition Information]({nutrition_link})\n\n"
            answer += "Would you like to know about another product?"

            steps.append(f"✅ Displayed details for: {name}")
            state['answer'] = answer
            state['reasoning_steps'] = steps
            update_conversation_context(session_id, question, answer, [product])
            return state

        state['answer'] = "I couldn't find that product. Please search for products first."
        return state

    # ── Ask which product ──────────────────────────────────────────────────
    if state.get('intent') == 'ask_which_product':
        state['answer'] = state.get('direct_response', "Which product would you like more details about?")
        return state

    # ── Direct response ────────────────────────────────────────────────────
    if state.get('direct_response'):
        steps.append("✅ Using direct response")
        state['answer'] = state['direct_response']
        state['reasoning_steps'] = steps
        return state

    # ── Instruction result ─────────────────────────────────────────────────
    if state.get('has_instructions') and state.get('instruction_data'):
        product      = state['instruction_data']
        product_name = product.get('name', 'this product')
        instructions = product.get('instructions', '')
        price        = product.get('price', '')

        if instructions:
            answer = f"📖 **How to make {product_name}**\n\n{instructions}\n\n**Price:** {price}\n\nHappy baking! 🍰"
        else:
            answer = (
                f"📖 **{product_name}**\n\nI don't have specific instructions for this product in my database.\n"
                f"Please check the product package or the King Arthur Baking website.\n\n**Price:** {price}"
            )

        steps.append("✅ Used direct instructions from database")
        state['answer'] = answer
        state['reasoning_steps'] = steps
        return state

    # ── Image request ──────────────────────────────────────────────────────
    # FIX: reads 'search_results' not 'found_products'
    search_results = state.get('search_results', [])

    if state.get('intent') == 'image_request' and search_results:
        products = search_results[:3]
        answer = f"**🖼️ Images for {len(products)} products:**\n\n"
        for i, product in enumerate(products, 1):
            name       = product.get('name', 'Unknown')
            price      = product.get('price', 'N/A')
            image_list = product.get('image_list', [])
            answer += f"**{i}. {name}** - {price}\n\n"
            if image_list:
                for img_url in image_list[:2]:
                    answer += f'<img src="{img_url}" width="200" style="border-radius:10px;margin:5px;">\n\n'
            else:
                answer += "📷 No image available.\n\n"
            answer += f"{'─'*40}\n\n"
        answer += "\nWould you like more information about any of these products?"

        steps.append("🖼️ Displayed product images")
        state['answer'] = answer
        state['reasoning_steps'] = steps
        update_conversation_context(session_id, question, answer, products)
        return state

    # ── Products found (sql / keyword / vector / hybrid) ──────────────────
    # FIX: reads 'search_results' not 'found_products'
    if search_results:
        products = search_results
        search_method = state.get('search_method', '')

        # Off-topic guard: only structured SQL queries (price/rating/count) are
        # guaranteed product-intent. For every OTHER method, if the question
        # isn't baking-related AND shares no product-name keyword with the hits,
        # do NOT recommend products — give a flexible answer instead. (Search,
        # especially semantic, returns the k nearest mixes even for unrelated
        # questions, which is what caused off-topic product recommendations.)
        explicit_product_query = (search_method == 'sql')
        if not explicit_product_query:
            on_topic = (_is_baking_related(question, llm, ollama_available)
                        or _shares_keyword(question, products))
            if not on_topic:
                steps.append("🚫 Off-topic query — gave flexible answer instead of products")
                answer = _flexible_answer(question, on_topic=False, llm=llm,
                                          ollama_available=ollama_available)
                _remember_learned(question, answer)
                state['answer'] = answer
                state['reasoning_steps'] = steps
                update_conversation_context(session_id, question, answer, [])
                return state

        # Show filter summary for SQL results
        filters = state.get('filters_applied', {})
        filter_note = ""
        if filters and search_method == 'sql':
            parts = []
            if filters.get('conditions'):
                parts.append(f"filters: {', '.join(filters['conditions'])}")
            if filters.get('order'):
                parts.append(f"sorted by: {filters['order'].replace('ORDER BY ','')}")
            if parts:
                filter_note = f"\n*({'; '.join(parts)})*\n\n"

        answer = f"**Found {len(products)} product(s):**{filter_note}\n\n"

        for i, p in enumerate(products, 1):
            name      = p.get('name', 'Unknown')
            price     = p.get('price', 'N/A')
            match_pct = p.get('match_percentage', 0)
            rating    = p.get('rating', '')

            answer += f"{i}. **{name}** - {price}"
            if rating:
                answer += f" ⭐ {rating}"
            answer += "\n"
            if match_pct > 0:
                answer += f"   📊 Match: {match_pct}%\n"
            if p.get('description'):
                answer += f"   📝 {p['description'][:150]}...\n"
            answer += "\n"

        answer += (
            "\nWould you like more details about any specific product? You can:\n"
            "• Type the number (1, 2, 3)\n"
            "• Say 'tell me about [product name]'\n"
            "• Ask 'show me image of [product name]'"
        )

        steps.append(f"✅ Listed {len(products)} products (method: {search_method})")
        _track_patterns(question, products)   # learning: query → products
        state['answer'] = answer
        state['reasoning_steps'] = steps
        update_conversation_context(session_id, question, answer, products)
        return state

    # ── No relevant product found → flexible LLM answer (NEVER dump catalog) ─
    # If the question is baking-related, answer it helpfully with the LLM.
    # If it's off-topic, politely steer back to baking — no product recommendations.
    on_topic = _is_baking_related(question, llm, ollama_available)
    answer = _flexible_answer(question, on_topic, llm, ollama_available)
    _remember_learned(question, answer)   # learning: persist informational answer

    steps.append(
        "✅ Flexible LLM answer (no relevant product match)"
        if on_topic else "🚫 Off-topic — redirected to baking topics"
    )
    state['answer'] = answer
    state['reasoning_steps'] = steps
    update_conversation_context(session_id, question, answer, [])
    return state