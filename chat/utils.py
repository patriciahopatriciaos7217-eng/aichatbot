"""
Utility functions for the agent
"""
import os
import logging

logger = logging.getLogger(__name__)


def setup_environment():
    """Configure environment variables for proxy bypass"""
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    logger.info("✅ Environment configured (proxy bypassed)")


def init_ollama():
    """Initialize Ollama with fallback models"""
    import subprocess
    
    # Try different models in order of preference
    models_to_try = ['phi3', 'llama3.2', 'llama3.2:1b', 'mistral', 'llama2']
    
    for model in models_to_try:
        try:
            # Check if model exists
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if model in result.stdout:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(
                    model=model,
                    temperature=0.5,
                    num_predict=512
                )
                print(f"✅ Ollama initialized with model: {model}")
                return llm, True
        except:
            continue
    
    print("⚠️ No Ollama models found. Please run: ollama pull llama3.2")
    return None, False


def get_database_functions():
    """Import database functions with fallback"""
    try:
        from database.dataManager import hybrid_search, search_sqlite, get_all_products, get_product_count
        logger.info("✅ Database functions loaded from database")
        return hybrid_search, search_sqlite, get_all_products, get_product_count
    except ImportError:
        try:
            from database.dataManager import hybrid_search, search_sqlite, get_all_products, get_product_count
            logger.info("✅ Database functions loaded from database.dataManager")
            return hybrid_search, search_sqlite, get_all_products, get_product_count
        except ImportError:
            # Mock functions for testing
            logger.warning("⚠️ Database functions not found, using mocks")
            
            def hybrid_search(q, k=5):
                return []
            
            def search_sqlite(c, v):
                return []
            
            def get_all_products():
                return []
            
            def get_product_count():
                return 0
            
            return hybrid_search, search_sqlite, get_all_products, get_product_count


def test_ollama_connection():
    """Test if Ollama is responding"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', 'unknown') for m in models]
            logger.info(f"✅ Ollama running with models: {model_names}")
            return True, model_names
        return False, []
    except Exception as e:
        logger.warning(f"⚠️ Ollama connection failed: {e}")
        return False, []