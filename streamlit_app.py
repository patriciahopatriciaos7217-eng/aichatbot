import streamlit as st
import subprocess
import sys
import os
from dotenv import load_dotenv
import json
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import atexit
from database.dataManager import close_connection
import re
import uuid
import logging

logger = logging.getLogger(__name__)

# Initialize session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

atexit.register(close_connection)

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
try:
    from database.dataManager import init_sqlite, init_chromadb, store_products, get_product_count, hybrid_search, save_feedback, search_products_smart
except ImportError:
    from database.dataManager import init_sqlite, init_chromadb, store_products, get_product_count, hybrid_search, save_feedback, search_products_smart

from chat.agent import ask_agent, test_ollama

load_dotenv()

# Page config
st.set_page_config(
    page_title="King Arthur Baking Assistant",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Product card styling */
    .product-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff6b35;
        word-break: break-all;
    }
    
    /* Match badge */
    .match-badge {
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ff6b35;
        border-radius: 10px;
    }
    
    /* Button hover */
    .stButton button {
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Chat message */
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .slider-image {
            max-width: 100%;
        }
    }
    
    /* Modal overlay */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .modal-content {
        background: white;
        border-radius: 20px;
        max-width: 800px;
        width: 90%;
        max-height: 90%;
        overflow-y: auto;
        padding: 20px;
        position: relative;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE SLIDER COMPONENT
# ============================================================

import hashlib

def image_slider(image_list: list, product_name: str = ""):
    """
    Working image slider using st.components.v1.html()
    JavaScript executes properly in its own iframe
    """
    if not image_list:
        st.info("📷 No images available for this product")
        return
    
    # Generate unique ID
    slider_id = hashlib.md5(f"{product_name}{len(image_list)}".encode()).hexdigest()[:8]
    
    # Build slides HTML
    slides_html = ""
    for idx, img_url in enumerate(image_list):
        display = "block" if idx == 0 else "none"
        slides_html += f"""
        <div class="slide" id="slide_{slider_id}_{idx}" style="display: {display}; text-align: center;">
            <img src="{img_url}" style="max-width: 100%; width: 300px; height: auto; max-height: 400px; object-fit: contain; border-radius: 15px;">
        </div>
        """
    
    # Build dots HTML
    dots_html = ""
    for idx in range(len(image_list)):
        active_class = 'active' if idx == 0 else ''
        dots_html += f'<span class="dot" data-index="{idx}" {active_class}></span>'
    
    # Complete HTML component
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: system-ui, -apple-system, sans-serif;
                background: transparent;
                padding: 10px;
            }}
            .slider-container {{
                position: relative;
                background: #f5f5f5;
                border-radius: 15px;
                min-height: 350px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .slide {{
                width: 100%;
                padding: 20px;
            }}
            .slide img {{
                max-width: 100%;
                max-height: 350px;
                object-fit: contain;
                border-radius: 15px;
            }}
            .nav-btn {{
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                background: #ff6b35;
                border: none;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 20px;
                transition: all 0.3s ease;
                z-index: 10;
            }}
            .nav-btn:hover {{
                background: #e55a2b;
                transform: translateY(-50%) scale(1.05);
            }}
            .prev-btn {{
                left: 10px;
            }}
            .next-btn {{
                right: 10px;
            }}
            .dots-container {{
                text-align: center;
                margin: 15px 0 10px 0;
            }}
            .dot {{
                display: inline-block;
                width: 10px;
                height: 10px;
                margin: 0 5px;
                background-color: #bbb;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .dot.active {{
                background-color: #ff6b35;
                width: 12px;
                height: 12px;
            }}
            .counter {{
                text-align: center;
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="slider-container">
            <button class="nav-btn prev-btn" id="prevBtn_{slider_id}">❮</button>
            {slides_html}
            <button class="nav-btn next-btn" id="nextBtn_{slider_id}">❯</button>
        </div>
        <div class="dots-container" id="dots_{slider_id}">
            {dots_html}
        </div>
        <div class="counter">
            📸 {product_name} - <span id="counter_{slider_id}">1</span> / {len(image_list)}
        </div>
        
        <script>
            (function() {{
                const totalSlides = {len(image_list)};
                let currentIndex = 0;
                
                // Get elements
                const slides = [];
                for (let i = 0; i < totalSlides; i++) {{
                    const slide = document.getElementById('slide_{slider_id}_' + i);
                    if (slide) slides.push(slide);
                }}
                
                const dots = document.querySelectorAll('#dots_{slider_id} .dot');
                const counterSpan = document.getElementById('counter_{slider_id}');
                const prevBtn = document.getElementById('prevBtn_{slider_id}');
                const nextBtn = document.getElementById('nextBtn_{slider_id}');
                
                function showSlide(index) {{
                    // Hide all slides
                    slides.forEach(slide => slide.style.display = 'none');
                    // Remove active class from dots
                    dots.forEach(dot => dot.classList.remove('active'));
                    
                    // Calculate new index (wrap around)
                    let newIndex = index;
                    if (newIndex < 0) newIndex = totalSlides - 1;
                    if (newIndex >= totalSlides) newIndex = 0;
                    
                    // Show current slide
                    if (slides[newIndex]) slides[newIndex].style.display = 'block';
                    if (dots[newIndex]) dots[newIndex].classList.add('active');
                    if (counterSpan) counterSpan.innerText = newIndex + 1;
                    
                    currentIndex = newIndex;
                }}
                
                // Event listeners
                if (prevBtn) prevBtn.onclick = function() {{ showSlide(currentIndex - 1); }};
                if (nextBtn) nextBtn.onclick = function() {{ showSlide(currentIndex + 1); }};
                
                // Dot click handlers
                dots.forEach((dot, i) => {{
                    dot.onclick = function() {{ showSlide(i); }};
                }});
                
                // Keyboard navigation
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'ArrowLeft') {{
                        showSlide(currentIndex - 1);
                        e.preventDefault();
                    }} else if (e.key === 'ArrowRight') {{
                        showSlide(currentIndex + 1);
                        e.preventDefault();
                    }}
                }});
            }})();
        </script>
    </body>
    </html>
    """
    
    # Use components.v1.html() instead of markdown
    st.components.v1.html(html_code, height=500)


# ============================================================
# PRODUCT DETAILS MODAL (Fixed - No @st.dialog)
# ============================================================

def show_product_details(product: dict):
    """Display product details in an expander or modal window"""
    
    name = product.get('name', 'Product Details')
    price = product.get('price', 'N/A')
    description = product.get('description', 'No description')
    ingredients = product.get('ingredients', 'No ingredients listed')
    contains = product.get('contains', 'No contains')
    image_list = product.get('image_list', [])
    instructions = product.get('instructions', '')
    nutrition_link = product.get('nutrition_link', '')
    url = product.get('url', '')
    review = product.get('review', '')
    product_code = product.get('product_code', '')
    details = product.get('details', '')
    
    # Create an expander for product details (this acts like a modal)
    with st.expander(f"🔍 {name} - Click to view details", expanded=True):
        # Product header
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"## 🍰 {name}")
            
            # Product metadata
            if product_code:
                st.markdown(f"**🆔 Code:** {product_code}")
                
            st.markdown(f"**💰 Price:** {price}")
            
            if review:
                st.markdown(f"**⭐ Reviews:** {review}")
                
            if details:
                st.markdown(f"**🍔 Detail Info:**")
                st.markdown(f"{details}")
        
        with col2:
            if image_list and len(image_list) > 0:
                st.image(image_list[0], use_container_width=True, caption=name[:30])
        
        st.divider()
        
        # Image slider (if multiple images)
        if len(image_list) > 1:
            st.markdown("### 📸 Product Images")
            image_slider(image_list, name)
        elif len(image_list) == 1:
            st.markdown("### 📸 Product Image")
            st.image(image_list[0], use_container_width=True)
        
        st.divider()
        
        # Product details tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Description", "🥗 Ingredients", "⚠️ Contains", "👩‍🍳 Instructions"])
        
        with tab1:
            st.write(description)
        
        with tab2:
            st.write(ingredients)
        
        with tab3:
            st.write(contains)
        
        with tab4:
            if instructions:
                st.write(instructions)
            else:
                st.info("No specific instructions available. Please check the product package.")
        
        # Nutrition link
        if nutrition_link:
            st.markdown("---")
            st.markdown(f"### 📊 Nutrition Information")
            st.markdown(f"[Click here to view full nutrition details]({nutrition_link})")
            
        if url:
            st.markdown("---")
            st.markdown(f"### 🔗 Product Link")
            st.markdown(f"[Click here to view on website]({url})")


def show_product_dialog(product: dict):
    """Display product details in a dialog-like container"""
    # Store selected product in session state
    st.session_state.selected_product = product
    st.session_state.show_product_modal = True


def render_product_modal():
    """Render the product modal if active"""
    if st.session_state.get('show_product_modal', False) and st.session_state.get('selected_product'):
        product = st.session_state.selected_product
        
        # Create a container that looks like a modal
        with st.container():
            st.markdown("---")
            st.markdown("## 📦 Product Details")
            
            # Close button
            col_close, _ = st.columns([1, 10])
            with col_close:
                if st.button("❌ Close", key="close_modal"):
                    st.session_state.show_product_modal = False
                    st.session_state.selected_product = None
                    st.rerun()
            
            st.markdown("---")
            
            # Show product details
            show_product_details(product)
            
            st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_product_card(product: dict, index: int):
    """Display a single product card with image and details"""
    name = product.get('name', 'Unknown')
    price = product.get('price', 'N/A')
    description = product.get('description', 'No description')[:150]
    match_pct = product.get('match_percentage', 0)
    image_list = product.get('image_list', [])
    
    # Create columns for image and text
    col_img, col_text = st.columns([1, 3])
    
    with col_img:
        if image_list and len(image_list) > 0:
            st.image(image_list[0], use_container_width=True, caption=name[:20])
        else:
            st.markdown("📷 No image")
    
    with col_text:
        st.markdown(f"**{index}. {name}** - {price}")
        if match_pct > 0:
            st.markdown(f'<span class="match-badge">🎯 Match: {match_pct}%</span>', unsafe_allow_html=True)
        st.caption(description)
    
    return name


def render_response_with_images(response: str):
    """Render response text with embedded images"""
    if '<img' in response:
        parts = re.split(r'(<img[^>]+>)', response)
        for part in parts:
            if part.startswith('<img'):
                img_match = re.search(r'src=["\']([^"\']+)["\']', part)
                if img_match:
                    st.image(img_match.group(1), use_container_width=True)
            else:
                st.markdown(part)
    else:
        st.markdown(response)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def get_all_mixes():
    """Get all mixes from JSON file"""
    products = []
    try:
        with open("./data/detailInfo.json", "r", encoding="utf-8") as f:
            products = json.load(f)
    except FileNotFoundError:
        st.warning("detailInfo.json not found")
    except Exception as e:
        st.error(f"Error loading detailInfo.json: {e}")
    return products


def get_sample_data():
    """Get sample data from JSON file"""
    data = []
    try:
        with open("./data/detailInfo.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning("products.json not found")
    except Exception as e:
        st.error(f"Error loading products.json: {e}")
    return data


def check_ollama_status():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return True, "Ollama is running"
        else:
            return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed. Please install from https://ollama.ai"
    except subprocess.TimeoutExpired:
        return False, "Ollama timed out. Please start with 'ollama serve'"
    except Exception as e:
        return False, f"Error: {str(e)}"


def initialize_system():
    """Initialize database and load data if needed"""
    try:
        init_sqlite()
        init_chromadb()
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return
    
    count = get_product_count()
    
    if count == 0:
        st.warning("📊 Database empty. Loading sample data for testing...")
        products = get_sample_data()
        if products:
            store_products(products)
            st.success(f"✅ Loaded {len(products)} sample products!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Scrape Real Data from Website"):
                with st.spinner("Scraping..."):
                    real_products = get_all_mixes()
                    if real_products:
                        store_products(real_products)
                        st.success(f"✅ Loaded {len(real_products)} real products!")
                        st.rerun()
                    else:
                        st.error("Scraping failed.")
        with col2:
            if st.button("Use Sample Data Only"):
                st.success("Using sample data.")
    else:
        st.success(f"✅ Database ready with {count} products")


def plot_agent_graph():
    """Visualize the agent graph"""
    G = nx.DiGraph()
    nodes = ["User Query", "Analyze Query", "Retrieve Info", "Generate Reasoning", "Generate Answer", "Check Quality"]
    edges = [
        ("User Query", "Analyze Query"),
        ("Analyze Query", "Retrieve Info"),
        ("Retrieve Info", "Generate Reasoning"),
        ("Generate Reasoning", "Generate Answer"),
        ("Generate Answer", "Check Quality")
    ]
    G.add_edges_from(edges)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    node_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, arrowstyle='->', arrowsize=20, ax=ax)
    
    plt.title("LangGraph Agent Architecture", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = []
if 'product_count' not in st.session_state:
    st.session_state.product_count = 0
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""
if 'show_product_modal' not in st.session_state:
    st.session_state.show_product_modal = False
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None


# ============================================================
# MAIN UI
# ============================================================

# Check Ollama status
ollama_ok, ollama_msg = check_ollama_status()

# Initialize system on first run
if not st.session_state.initialized:
    if ollama_ok:
        initialize_system()
        st.session_state.initialized = True
        st.session_state.product_count = get_product_count()
    else:
        st.session_state.initialized = False

# Title
st.title("🍞 King Arthur Baking Assistant")
st.caption("Your AI-powered guide to perfect baking | Powered by Ollama + LangGraph")

# Ollama status warning
if not ollama_ok:
    st.error(f"⚠️ Ollama is not running! {ollama_msg}")
    st.info("Run `ollama serve` in terminal to start Ollama")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🤖 Assistant Info")
    st.markdown("""
    This assistant helps you discover King Arthur Baking products.
    
    **Features:**
    - 🔍 Search baking mixes
    - 📝 Check ingredients
    - 🖼️ View product images
    - 👩‍🍳 Get baking instructions
    - 🥗 Dietary filtering
    """)
    
    st.divider()
    
    st.header("📊 Agent Graph")
    if st.button("Show Architecture", use_container_width=True):
        plot_agent_graph()
    
    st.header("📦 Database")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Stats", use_container_width=True):
            st.session_state.product_count = get_product_count()
            st.rerun()
    with col2:
        st.metric("Products", st.session_state.product_count)
    
    st.divider()
    
    st.header("🔄 Data")
    if st.button("Load Sample Data", use_container_width=True):
        with st.spinner("Loading..."):
            products = get_sample_data()
            if products:
                store_products(products)
                st.success(f"✅ Loaded {len(products)} samples")
                st.session_state.product_count = get_product_count()
                st.rerun()
    
    if st.button("Scrape Real Data", use_container_width=True):
        with st.spinner("Scraping..."):
            products = get_all_mixes()
            if products:
                store_products(products)
                st.success(f"✅ Loaded {len(products)} products")
                st.session_state.product_count = get_product_count()
                st.rerun()
    
    st.divider()
    
    st.header("ℹ️ Model Info")
    st.info("""
    **LLM:** llama3.2  
    **Embedding:** nomic-embed-text  
    **Search:** SQLite + ChromaDB  
    **Framework:** LangGraph
    """)


# ============================================================
# CHAT INTERFACE
# ============================================================

chat_col1, chat_col2 = st.columns([2, 1])

with chat_col1:
    st.header("💬 Chat")
    
    # Chat input
    user_question = st.chat_input("Ask about baking mixes...")
    
    if user_question:
        with st.spinner("🧠 Thinking..."):
            try:
                response = ask_agent(user_question, session_id=st.session_state.session_id)
                st.session_state.chat_history.append({
                    "user": user_question,
                    "bot": response.get('answer', 'No response'),
                    "reasoning": response.get('reasoning_steps', []),
                    "products": response.get('suggested_products', []),
                    "products_data": response.get('products_data', [])
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display chat history
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.chat_message("user"):
            st.write(chat['user'])
        
        with st.chat_message("assistant"):
            # Render response with images
            render_response_with_images(chat['bot'])
            
            # Display product cards if available
            if chat.get('products_data'):
                st.markdown("---")
                st.markdown("### 📦 Related Products")
                for p_idx, product in enumerate(chat['products_data'][:5], 1):
                    with st.container():
                        display_product_card(product, p_idx)
                        if st.button(f"🔍 View Details", key=f"details_{idx}_{p_idx}"):
                            show_product_dialog(product)
                            st.rerun()
            
            # Feedback buttons
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("👍 Helpful", key=f"helpful_{idx}"):
                    save_feedback(chat['user'], chat['bot'], 1, None)
                    st.toast("Thanks for your feedback! 🎉")
            with col_fb2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{idx}"):
                    correction = st.text_input("Better answer:", key=f"correction_{idx}")
                    if correction:
                        save_feedback(chat['user'], chat['bot'], 0, correction)
                        st.toast("Thanks! I'll learn from this! 📚")

with chat_col2:
    st.header("🧠 Agent Reasoning")
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        if latest.get('reasoning'):
            for i, step in enumerate(latest['reasoning'][-4:], 1):
                with st.expander(f"Step {i}"):
                    st.caption(step[:500])
    
    st.divider()
    
    st.header("📚 Quick Questions")
    quick_questions = [
        "What gluten-free mixes do you have?",
        "Show me brownie mix in detail",
        "Show me pizza crust mixes",
        "Does the cake mix contain dairy?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state.selected_question = q
            st.rerun()
    
    if st.session_state.selected_question:
        st.info(f"Selected: {st.session_state.selected_question}")
        if st.button("Clear"):
            st.session_state.selected_question = ""
            st.rerun()
    
    st.divider()
    
    st.header("💡 Tips")
    st.markdown("""
    - Be specific about dietary needs
    - Ask for product details or images
    - Use quick questions for examples
    """)


# ============================================================
# RENDER PRODUCT MODAL
# ============================================================

render_product_modal()


# ============================================================
# SELECTED QUESTION HANDLER
# ============================================================

if st.session_state.selected_question:
    with st.spinner("🧠 Thinking..."):
        try:
            response = ask_agent(st.session_state.selected_question, session_id=st.session_state.session_id)
            st.session_state.chat_history.append({
                "user": st.session_state.selected_question,
                "bot": response.get('answer', 'No response'),
                "reasoning": response.get('reasoning_steps', []),
                "products": response.get('suggested_products', []),
                "products_data": response.get('products_data', [])
            })
            st.session_state.selected_question = ""
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.selected_question = ""


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🍞 100% Local AI | Ollama + LangGraph + ChromaDB | No data leaves your computer</p>", 
    unsafe_allow_html=True
)