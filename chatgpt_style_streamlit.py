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

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }

    /* Product card styling - FIXED */
    .product-card {
        background-color: #f9f9f9;
        padding: 16px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 4px solid #ff6b35;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        width: 100%;
        box-sizing: border-box;
    }

    .product-card-image {
        flex-shrink: 0;
        width: 120px;
        height: 120px;
        border-radius: 8px;
        overflow: hidden;
        background: #fff;
    }

    .product-card-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .product-card-content {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .product-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }

    .product-card-price {
        font-size: 14px;
        color: #ff6b35;
        font-weight: 600;
    }

    .product-card-description {
        font-size: 13px;
        color: #666;
        line-height: 1.4;
        margin: 0;
    }

    .product-card-match {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        width: fit-content;
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
        .product-card {
            flex-direction: column;
        }
        .product-card-image {
            width: 100%;
            height: auto;
            min-height: 150px;
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

    /* ChatGPT-like: add space so messages aren't hidden under the
       native sticky chat input (do NOT re-position st.chat_input;
       Streamlit already pins it to the bottom of the main area). */
    .main .block-container {
        padding-bottom: 6rem !important;
    }

    /* Custom avatar background tint */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f7f7f8;
    }
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff;
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
        <div class="slide" id="slide-{slider_id}-{idx}" style="display: {display};">
            <img src="{img_url}" style="width:100%; max-height:350px; object-fit:contain; border-radius:8px;" />
        </div>
        """

    # Build dots HTML
    dots_html = ""
    for idx in range(len(image_list)):
        active_class = 'active' if idx == 0 else ''
        dots_html += f'<span class="dot {active_class}" onclick="showSlide(\'{slider_id}\', {idx})"></span>'

    # Complete HTML component
    html_code = f"""
    <style>
        .slider-container-{slider_id} {{
            position: relative;
            max-width: 100%;
            margin: auto;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
        }}
        .slide {{
            display: none;
        }}
        .slide:first-child {{
            display: block;
        }}
        .prev, .next {{
            cursor: pointer;
            position: absolute;
            top: 50%;
            width: auto;
            padding: 12px;
            margin-top: -22px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            transition: 0.3s ease;
            border-radius: 0 3px 3px 0;
            user-select: none;
            background-color: rgba(0,0,0,0.3);
        }}
        .next {{
            right: 0;
            border-radius: 3px 0 0 3px;
        }}
        .prev:hover, .next:hover {{
            background-color: rgba(0,0,0,0.6);
        }}
        .dots-container {{
            text-align: center;
            padding: 10px;
        }}
        .dot {{
            cursor: pointer;
            height: 12px;
            width: 12px;
            margin: 0 4px;
            background-color: #bbb;
            border-radius: 50%;
            display: inline-block;
            transition: background-color 0.3s ease;
        }}
        .dot.active, .dot:hover {{
            background-color: #ff6b35;
        }}
        .caption {{
            text-align: center;
            padding: 8px;
            color: #666;
            font-size: 14px;
        }}
    </style>

    <div class="slider-container-{slider_id}">
        <div class="slides-wrapper">
            {slides_html}
        </div>

        <a class="prev" onclick="changeSlide('{slider_id}', -1)">&#10094;</a>
        <a class="next" onclick="changeSlide('{slider_id}', 1)">&#10095;</a>

        <div class="dots-container">
            {dots_html}
        </div>

        <div class="caption">
            📸 {product_name} - 1 / {len(image_list)}
        </div>
    </div>

    <script>
        var slideIndices = {{}};
        slideIndices['{slider_id}'] = 0;

        function changeSlide(sliderId, n) {{
            showSlide(sliderId, slideIndices[sliderId] + n);
        }}

        function showSlide(sliderId, n) {{
            var i;
            var container = document.querySelector('.slider-container-' + sliderId);
            var slides = container.getElementsByClassName('slide');
            var dots = container.getElementsByClassName('dot');

            if (n >= slides.length) {{ slideIndices[sliderId] = 0 }}
            else if (n < 0) {{ slideIndices[sliderId] = slides.length - 1 }}
            else {{ slideIndices[sliderId] = n }}

            for (i = 0; i < slides.length; i++) {{
                slides[i].style.display = "none";
            }}
            for (i = 0; i < dots.length; i++) {{
                dots[i].className = dots[i].className.replace(" active", "");
            }}

            slides[slideIndices[sliderId]].style.display = "block";
            dots[slideIndices[sliderId]].className += " active";
        }}
    </script>
    """

    # Use components.v1.html() instead of markdown
    st.components.v1.html(html_code, height=500)


# ============================================================
# PRODUCT DETAILS MODAL (Fixed - No @st.dialog)
# ============================================================

def show_product_details(product: dict):
    """Display product details in an expander or modal window"""
    print(ingredients)
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
    st.session_state.selected_product = product
    st.session_state.show_product_modal = True
    render_product_modal()


def render_product_modal():
    """Render the product modal if active"""
    if st.session_state.get('show_product_modal', False) and st.session_state.get('selected_product'):
        product = st.session_state.selected_product

        with st.container():
            st.markdown("---")
            st.markdown("## 📦 Product Details")

            col_close, _ = st.columns([1, 10])
            with col_close:
                if st.button("❌ Close", key="close_modal"):
                    st.session_state.show_product_modal = False
                    st.session_state.selected_product = None
                    st.rerun()

            st.markdown("---")

            show_product_details(product)

            st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_product_card(product: dict, index: int):
    """Display a single product card with improved styling"""
    name = product.get('name', 'Unknown')
    price = product.get('price', 'N/A')
    description = product.get('description', 'No description')
    match_pct = product.get('match_percentage', 0)
    image_list = product.get('image_list', [])
    print('image_list', product)

    # Get first image or placeholder
    img_url = image_list[0] if image_list else None

    # Build the HTML card
    card_html = f"""
    <div class="product-card">
        <div class="product-card-image">
            {'<img src="' + img_url + '" />' if img_url else '<div style="background: #eee; display: flex; align-items: center; justify-content: center; height: 100%; color: #999;">📷 No image</div>'}
        </div>
        <div class="product-card-content">
            <div class="product-card-title">{index}. {name}</div>
            <div class="product-card-price">{price}</div>
            {f'<div class="product-card-match">🎯 Match: {match_pct}%</div>' if match_pct > 0 else ''}
            <div class="product-card-description">{description}</div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_response_with_images(response: str):
    """Render response text with embedded images"""
    if '<img' in response:
        parts = re.split(r'(<img[^>]+>)', response)
        for part in parts:
            if part.startswith('<img'):
                src_match = re.search(r'src=["\']([^"\']+)["\']', part)
                if src_match:
                    st.image(src_match.group(1), use_container_width=True)
            else:
                st.markdown(part)
    else:
        st.markdown(response)


# ============================================================
# CHATGPT-STYLE UI
# ============================================================

def render_chat_interface():
    """Render ChatGPT-style chat interface with messages on top and input at bottom"""

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input at the bottom
    if prompt := st.chat_input("Ask me anything about baking..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_agent(prompt)

                    # Extract the answer text (handle both dict and string)
                    if isinstance(response, dict):
                        answer = response.get("answer", "No answer provided.")
                        products_data = response.get("products_data", [])
                    else:
                        answer = str(response)
                        products_data = []

                    # Display the answer
                    st.markdown(answer)

                    # Display product cards if any
                    if products_data:
                        st.markdown("---")
                        st.markdown("### 📦 Related Products")
                        for idx, product in enumerate(products_data, 1):
                            with st.container():
                                display_product_card(product, idx)
                                if st.button(f"🔍 View Details", key=f"details_{idx}"):
                                    show_product_dialog(product)

                    # Add assistant response to chat history (store only the answer text)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Title
    st.markdown("""
    <div class="main-header">
        <h1>🍞 King Arthur Baking Assistant</h1>
        <p>Your AI-powered baking companion</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🍞 King Arthur Baking Assistant")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Search Products", "🧠 LangGraph", "📊 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("🍞 100% Local AI | Ollama + LangGraph + ChromaDB | No data leaves your computer")

    # Main content area
    if page == "💬 Chat":
        render_chat_interface()

    elif page == "🔍 Search Products":
        st.markdown("## 🔍 Search Products")
        search_query = st.text_input("Enter your search query:", placeholder="e.g., organic flour, bread yeast...")
        col1, col2 = st.columns([1, 1])
        with col1:
            search_btn = st.button("🔍 Search", use_container_width=True)
        with col2:
            smart_search_btn = st.button("🧠 Smart Search", use_container_width=True)
        if search_btn and search_query:
            with st.spinner("Searching..."):
                try:
                    results = hybrid_search(search_query)
                    if results:
                        st.success(f"Found {len(results)} products")
                        for idx, product in enumerate(results, 1):
                            display_product_card(product, idx)
                            if st.button(f"🔍 View Details", key=f"search_details_{idx}"):
                                print("detail", product)
                                show_product_dialog(product)
                    else:
                        st.info("No products found matching your query.")
                except Exception as e:
                    st.error(f"Search error: {e}")
        if smart_search_btn and search_query:
            with st.spinner("Smart searching..."):
                try:
                    results = search_products_smart(search_query)
                    if results:
                        st.success(f"Found {len(results)} products (smart search)")
                        for idx, product in enumerate(results, 1):
                            display_product_card(product, idx)
                            if st.button(f"🔍 View Details", key=f"smart_details_{idx}"):
                                show_product_dialog(product)
                    else:
                        st.info("No products found with smart search.")
                except Exception as e:
                    st.error(f"Smart search error: {e}")

    elif page == "🧠 LangGraph":
        render_langgraph_view()

    elif page == "📊 Analytics":
        st.markdown("## 📊 Analytics")
        try:
            count = get_product_count()
            st.metric("Total Products", count)
        except Exception as e:
            st.error(f"Could not load analytics: {e}")
        st.info("Analytics dashboard coming soon...")

    elif page == "⚙️ Settings":
        st.markdown("## ⚙️ Settings")
        st.markdown("### Database")
        if st.button("🔄 Initialize SQLite"):
            with st.spinner("Initializing..."):
                try:
                    init_sqlite()
                    st.success("SQLite initialized!")
                except Exception as e:
                    st.error(f"Error: {e}")
        if st.button("🔄 Initialize ChromaDB"):
            with st.spinner("Initializing..."):
                try:
                    init_chromadb()
                    st.success("ChromaDB initialized!")
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown("### AI Model")
        if st.button("🧪 Test Ollama"):
            with st.spinner("Testing..."):
                try:
                    result = test_ollama()
                    st.success(f"Ollama is working! Response: {result}")
                except Exception as e:
                    st.error(f"Ollama test failed: {e}")




# ============================================================
# LANGGRAPH VISUALIZATION
# ============================================================

def render_langgraph_view():
    """Render the LangGraph agent workflow as an interactive graph."""
    st.markdown("## 🧠 LangGraph Agent Workflow")
    st.caption("Visualization of the agent's reasoning graph (nodes = steps, edges = transitions).")

    # Try to pull the compiled graph from the agent module; fall back to a
    # static representation that mirrors the known pipeline.
    nodes = ["START", "classify_intent", "retrieve_products", "generate_answer", "format_response", "END"]
    edges = [
        ("START", "classify_intent"),
        ("classify_intent", "retrieve_products"),
        ("classify_intent", "generate_answer"),
        ("retrieve_products", "generate_answer"),
        ("generate_answer", "format_response"),
        ("format_response", "END"),
    ]

    try:
        from chat import agent as _agent
        g = getattr(_agent, "graph", None) or getattr(_agent, "compiled_graph", None)
        if g is not None and hasattr(g, "get_graph"):
            gv = g.get_graph()
            nodes = list(gv.nodes.keys())
            edges = [(e.source, e.target) for e in gv.edges]
    except Exception as e:
        st.info(f"Using fallback graph layout ({e}).")

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Plotly interactive figure
    edge_x, edge_y = [], []
    for src, dst in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none", mode="lines",
    )
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(size=28, color="#ff6b35", line=dict(width=2, color="#fff")),
        hoverinfo="text",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Nodes & Edges"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nodes**")
            for n in nodes:
                st.markdown(f"- `{n}`")
        with col2:
            st.markdown("**Edges**")
            for s, t in edges:
                st.markdown(f"- `{s}` → `{t}`")



if __name__ == "__main__":
    main()
