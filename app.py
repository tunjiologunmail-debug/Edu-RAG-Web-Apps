# app.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: The main Streamlit web application.
# Streamlit turns Python scripts into interactive web apps with no HTML/CSS needed.
#
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import os
import time
from dotenv import load_dotenv

# Import our custom modules
from src.retriever import load_vectorstore, retrieve_relevant_chunks, format_retrieved_chunks, get_sources_list
from src.llm_chain import build_rag_chain, generate_answer_streaming

load_dotenv()

# ── Page Configuration ─────────────────────────────────────────────────────────
# This must be the FIRST Streamlit command in the script
st.set_page_config(
    page_title="EduRAG — University Policy Assistant",
    page_icon="🎓",
    layout="wide",                    # Use the full browser width
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Make the main content area look cleaner */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2e86ab);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .source-box {
        background: #f0f4f8;
        border-left: 4px solid #2e86ab;
        padding: 0.75rem 1rem;
        border-radius: 0 5px 5px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Caching: Load expensive resources once ────────────────────────────────────
# @st.cache_resource tells Streamlit: "load this once and reuse it"
# Without this, the FAISS index and LLM would reload on EVERY interaction

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store once and cache it."""
    index_path = "data/faiss_index"

    if not os.path.exists(index_path):
        return None

    return load_vectorstore(index_path)


@st.cache_resource
def get_llm_chain():
    """Build the LLM chain once and cache it."""
    return build_rag_chain()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=EduRAG", width=200)
    st.markdown("## ⚙️ Settings")

    # Slider to control how many document chunks to retrieve
    top_k = st.slider(
        label="Number of sources to retrieve",
        min_value=2,
        max_value=8,
        value=4,
        help="More sources = broader context but slower response. 4 is recommended."
    )

    show_context = st.checkbox(
        label="Show retrieved document chunks",
        value=False,
        help="Useful for understanding HOW the answer was generated (great for demos!)"
    )

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    EduRAG answers questions about university policies and documents using
    **Retrieval-Augmented Generation (RAG)**.

    Unlike a standard chatbot, answers are grounded in **your institution's
    actual documents** — not AI guesswork.
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    This tool assists with document queries but may not reflect the most
    current policies. Always verify important decisions with your institution.
    """)


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 EduRAG</h1>
    <p>Ask questions about university policies, handbooks, and regulations</p>
</div>
""", unsafe_allow_html=True)


# ── Check if index exists ──────────────────────────────────────────────────────
vectorstore = get_vectorstore()

if vectorstore is None:
    # Show setup instructions if the index hasn't been built yet
    st.error("⚠️ No document index found!")
    st.info("""
    **To get started:**
    1. Add PDF files to the `data/pdfs/` folder
    2. Open a terminal in this folder
    3. Run: `python src/ingest.py`
    4. Refresh this page
    """)
    st.stop()  # Don't render anything else


# ── Example Questions ──────────────────────────────────────────────────────────
st.markdown("### 💡 Try asking...")

# Create clickable example question buttons
example_questions = [
    "What is the policy on academic misconduct?",
    "How do I apply for an extension on my assignment?",
    "What support is available for students with disabilities?",
    "How does the appeals process work?"
]

# Display buttons in a grid (2 columns)
col1, col2 = st.columns(2)
for i, question in enumerate(example_questions):
    col = col1 if i % 2 == 0 else col2
    if col.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
        # When a button is clicked, store the question in session state
        st.session_state.clicked_question = question


# ── Question Input ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask Your Question")

# Pre-fill the input if an example was clicked
default_question = st.session_state.get("clicked_question", "")

user_question = st.text_input(
    label="Type your question here:",
    value=default_question,
    placeholder="e.g. What happens if I miss an exam?",
    label_visibility="collapsed"
)

# Clear the clicked question so it doesn't persist
if "clicked_question" in st.session_state:
    del st.session_state.clicked_question

# Submit button
ask_button = st.button("🔍 Search Documents", type="primary", use_container_width=True)


# ── Answer Generation ──────────────────────────────────────────────────────────
if ask_button and user_question.strip():

    # Track response time (useful for your evaluation report)
    start_time = time.time()

    with st.spinner("📚 Searching university documents..."):
        # Step 1: Retrieve relevant chunks
        docs = retrieve_relevant_chunks(user_question, vectorstore, top_k=top_k)
        context = format_retrieved_chunks(docs)
        sources = get_sources_list(docs)

    retrieval_time = time.time() - start_time

    st.markdown("---")
    st.markdown("### 💡 Answer")

    # Create an answer container for streaming output
    answer_container = st.empty()

    # Step 2: Generate answer with streaming
    llm_chain = get_llm_chain()
    full_answer = ""

    # Stream the response token by token (feels faster to the user)
    with st.spinner("🤖 Generating answer..."):
        for chunk in generate_answer_streaming(user_question, context, llm_chain):
            full_answer += chunk
            answer_container.markdown(full_answer)

    total_time = time.time() - start_time

    # Step 3: Display sources
    st.markdown("### 📎 Sources Used")
    for source in sources:
        st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)

    # Show performance metrics (great for your evaluation report section)
    with st.expander("📊 Response Metrics"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Response Time", f"{total_time:.1f}s")
        col2.metric("Chunks Retrieved", len(docs))
        col3.metric("Sources Cited", len(sources))

    # Optionally show the raw retrieved chunks (for transparency demo)
    if show_context:
        with st.expander("🔬 View Retrieved Document Chunks (Debug Mode)"):
            st.markdown("""
            *These are the exact passages retrieved from your documents.
            The LLM's answer is based ONLY on this content.*
            """)
            st.text(context)

    # User feedback (simple thumbs up/down — good for evaluation)
    st.markdown("---")
    st.markdown("**Was this answer helpful?**")
    col1, col2, col3 = st.columns([1, 1, 4])
    col1.button("👍 Yes", key="thumbs_up")
    col2.button("👎 No", key="thumbs_down")


elif ask_button and not user_question.strip():
    st.warning("Please enter a question first.")


# ── Conversation History ───────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction,
# so we use st.session_state to persist the chat history
if "history" not in st.session_state:
    st.session_state.history = []