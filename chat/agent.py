"""
Main Agent Interface - Exports functions for Streamlit
"""
import os
import sys
import logging

# Fix proxy issues
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# Import from agent_graph
from .agent_graph import get_agent_graph, set_llm
from .utils import setup_environment, init_ollama, get_database_functions
from .agent_state import create_initial_state

# Setup
setup_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ollama
llm, OLLAMA_AVAILABLE = init_ollama()

# Set LLM for graph nodes
set_llm(llm, OLLAMA_AVAILABLE)

# Get graph
agent_graph = get_agent_graph()


# In chat/agent.py, update ask_agent function
def ask_agent(question: str, session_id: str = "default") -> dict:
    """
    Main function with session context
    """
    from .agent_nodes import (
        get_conversation_context, update_conversation_context, is_cacheable_question,
    )
    from database.dataManager import (
        get_learned_response, save_chat_history,
    )

    # Get context for this session
    context = get_conversation_context(session_id)
    last_products = context.get("last_search_results", [])
    print(question)

    # ── Learning fast-path ────────────────────────────────────────────────
    # If we've already learned an answer for this exact question, serve it
    # directly (consistent + fast). Only informational answers are ever stored
    # (see _remember_learned), so live product searches are never short-circuited.
    # Skip context-dependent queries (numbers/ordinals/follow-ups).
    try:
        learned = get_learned_response(question) if is_cacheable_question(question) else None
    except Exception as e:
        logger.error(f"get_learned_response failed: {e}")
        learned = None
    if learned:
        update_conversation_context(session_id, question, learned, [])
        try:
            save_chat_history(session_id, question, learned, intent="learned", product_count=0)
        except Exception as e:
            logger.error(f"save_chat_history failed: {e}")
        return {
            "answer": learned,
            "reasoning_steps": ["📚 Served a previously learned answer"],
            "suggested_products": [],
            "products_data": [],
            "product_count": 0,
            "intent": "learned",
        }

    # Create state with context
    initial_state = create_initial_state(question)
    initial_state['last_search_results'] = last_products
    initial_state['session_id'] = session_id

    # Run agent
    graph = get_agent_graph()

    try:
        final_state = graph.invoke(initial_state)

        # Update conversation context
        products = final_state.get('suggested_products', [])
        # NOTE: search nodes write to the unified 'search_results' key
        # (was 'found_products' before the migration — that key is never set).
        products_data = final_state.get('search_results', [])
        answer = final_state.get('answer', '')
        update_conversation_context(session_id, question, answer, products_data)

        # Persist the turn (chat history + learning signal).
        try:
            save_chat_history(
                session_id, question, answer,
                intent=final_state.get('intent', ''),
                product_count=final_state.get('product_count', 0),
            )
        except Exception as e:
            logger.error(f"save_chat_history failed: {e}")

        return {
            "answer": final_state.get("answer", "No response"),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "suggested_products": products,
            "products_data": products_data,
            "product_count": final_state.get("product_count", 0),
            "intent": final_state.get("intent", "unknown")
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "reasoning_steps": ["Error occurred"],
            "suggested_products": [],
            "products_data": [],
            "product_count": 0,
            "intent": "error"
        }

def test_ollama() -> tuple:
    """Test if Ollama is available"""
    if OLLAMA_AVAILABLE and llm:
        try:
            response = llm.invoke("Say OK")
            return True, "Ollama is working"
        except Exception as e:
            return False, f"Ollama error: {e}"
    return False, "Ollama not available. Run 'ollama serve' and install langchain-ollama"


def get_agent_graph_for_visualization():
    """Get agent graph for frontend visualization"""
    return agent_graph


# For direct testing
if __name__ == "__main__":
    print("Testing Agent")
    print("=" * 50)
    
    ok, msg = test_ollama()
    print(f"Ollama: {msg}")
    
    result = ask_agent("What gluten-free mixes do you have?")
    print(f"\nAnswer: {result['answer'][:200]}...")
    print(f"Intent: {result['intent']}")
    print(f"Products found: {result['product_count']}")