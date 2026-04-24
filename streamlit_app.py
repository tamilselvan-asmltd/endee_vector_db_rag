import streamlit as st
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure we can import from core/config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Silence Noisy Transformers Logs ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------------------

# --- Configure NLTK for Auto-Download & Offline Use ---
from config.settings import settings
os.environ["NLTK_DATA"] = settings.nltk_data_path
try:
    import nltk
    import os
    
    os.makedirs(settings.nltk_data_path, exist_ok=True)
    
    # Ensure offline paths are prioritized
    if settings.nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, settings.nltk_data_path)
        
    # Auto-download missing mandatory packages to the local unified folder
    required_pkgs = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, pkg_name in required_pkgs:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"[*] Downloading missing NLTK package: {pkg_name}...")
            nltk.download(pkg_name, download_dir=settings.nltk_data_path, quiet=True)
            
except ImportError:
    pass
# ------------------------------------------------------------------------

from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever
from core.generator import GenerationService
from main import ingest

# --- Page Configuration ---
st.set_page_config(
    page_title="Endee | AI Engineering Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Light Theme (White, Light Blue, Orange) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f9ff !important;
        border-right: 1px solid #e0f2fe;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    /* Headers & Branding - Industrial Orange */
    h1, h2, h3 {
        color: #f97316 !important;
        font-weight: 700 !important;
    }
    
    /* Chat Bubble Styling */
    /* Assistant Bubble (White/Gray) */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px;
        color: #1e293b !important;
    }
    
    /* User Bubble (Light Blue) */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e0f2fe !important;
        border: 1px solid #bae6fd !important;
        border-radius: 12px;
        color: #0c4a6e !important;
    }
    
    /* Buttons - Light Blue */
    .stButton button {
        background-color: #0ea5e9 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0284c7 !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Links - Light Blue */
    a {
        color: #0ea5e9 !important;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Input & Widgets */
    .stChatInput {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Status Messages */
    .stStatusWidget {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if "active_search_filter" not in st.session_state:
    st.session_state.active_search_filter = []

# --- Core RAG Logic ---
def initialize_services():
    """Initializes RAG services and caches them in session state."""
    try:
        embeddings = EmbeddingService()
        db = DatabaseService()
        index = db.get_index()
        
        # Use active search filter from session state
        retriever = HybridEndeeRetriever(
            index=index, 
            embedding_service=embeddings,
            base_filter=st.session_state.active_search_filter
        )
        generator = GenerationService(retriever)
        
        st.session_state.generator = generator
        st.session_state.system_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize AI Engine: {e}")

def save_uploaded_files(uploaded_files: List[Any]) -> List[str]:
    """Saves uploaded files to the 'docs/' directory."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = docs_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))
    return saved_paths

# --- Sidebar: Ingestion & Control ---
with st.sidebar:
    st.title("Knowledge Library")
    
    st.subheader("🔍 Search Filters")
    search_filter_input = st.text_input(
        "Active Search Filter",
        placeholder="e.g. dept=maint",
        help="Apply filters to your questions. Format: key=value",
        key="search_filter_input"
    )
    
    if st.button("Apply Search Filter"):
        if search_filter_input:
            try:
                new_filter = []
                pairs = [p.strip() for p in search_filter_input.split(",")]
                for pair in pairs:
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        new_filter.append({k.strip(): {"$eq": v.strip()}})
                
                if new_filter:
                    st.session_state.active_search_filter = new_filter
                    st.success(f"Filters applied: {search_filter_input}")
                    initialize_services()
                else:
                    st.error("Invalid format. Use key=value, key2=value2")
            except Exception as e:
                st.error(f"Filter error: {e}")
        else:
            st.session_state.active_search_filter = []
            st.info("Filters cleared.")
            initialize_services()

    st.divider()
    st.write("Upload PDF manuals to expand the AI's engineering knowledge.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Manuals",
        type="pdf",
        accept_multiple_files=True
    )
    
    recreate_index = st.checkbox("Fresh Start (Wipe Index)", value=False)
    
    metadata_input = st.text_input(
        "Metadata Filters (Optional)",
        placeholder="e.g. dept=maint, machine=cnc01",
        help="Custom tags to apply to these documents. Format: key=value, key2=value2"
    )
    
    if st.button("🚀 Synchronize Data", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select files before synchronizing.")
        else:
            with st.status("Ingesting Documents...", expanded=True) as status:
                saved_paths = save_uploaded_files(uploaded_files)
                for i, path in enumerate(saved_paths):
                    st.write(f"Processing ({i+1}/{len(saved_paths)}): {Path(path).name}")
                    run_recreate = recreate_index if i == 0 else False
                    
                    # Parse metadata filters
                    extra_meta = {}
                    if metadata_input:
                        try:
                            pairs = [p.strip() for p in metadata_input.split(",")]
                            for pair in pairs:
                                if "=" in pair:
                                    k, v = pair.split("=", 1)
                                    extra_meta[k.strip()] = v.strip()
                        except Exception as e:
                            st.error(f"Error parsing metadata filters: {e}")

                    try:
                        ingest(path, recreate=run_recreate, extra_metadata=extra_meta)
                    except Exception as e:
                        st.error(f"Error processing {Path(path).name}: {e}")
                
                status.update(label="Library Updated!", state="complete", expanded=False)
                st.success(f"Successfully processed {len(saved_paths)} manual(s).")
                initialize_services()

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("🛠️ Index Management"):
        st.write("Delete chunks based on metadata filters.")
        del_filter_input = st.text_input(
            "Delete Filter",
            placeholder="key=value",
            key="del_filter"
        )
        if st.button("🗑️ Delete Chunks", type="primary", use_container_width=True):
            if not del_filter_input:
                st.warning("Please provide a filter for deletion.")
            else:
                try:
                    new_del_filter = []
                    pairs = [p.strip() for p in del_filter_input.split(",")]
                    for pair in pairs:
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            new_del_filter.append({k.strip(): {"$eq": v.strip()}})
                    
                    if new_del_filter:
                        from main import delete_by_filter
                        delete_by_filter(new_del_filter)
                        st.success(f"Deletion successful for: {del_filter_input}")
                    else:
                        st.error("Invalid filter format. Use key=value, key2=value2")
                except Exception as e:
                    st.error(f"Deletion failed: {e}")

# --- Main UI ---
st.title("CNC Support Intelligence")
st.write("Expert guidance based on your engineering and maintenance documentation.")

# Initialize services
if not st.session_state.system_initialized:
    with st.spinner("Connecting to Engineering Intelligence Engine..."):
        initialize_services()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📌 Source References"):
                for source in message["sources"]:
                    st.markdown(f"**[{source['filename']}]({source['link']})** • Page {source['page']}")

# Chat Input & AI Workflow
if prompt := st.chat_input("Ask about setups, maintenance, or operations..."):
    # User Perspective
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Perspective
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if st.session_state.system_initialized:
            generator = st.session_state.generator
            
            try:
                # 1. Retrieval Phase
                with st.spinner("Analyzing manuals..."):
                    context_docs = generator.retriever.invoke(prompt)
                    retrieval_time = getattr(generator.retriever, "last_retrieval_time", 0.0)
                
                if context_docs:
                    st.toast(f"🔍 Found {len(context_docs)} relevant context points in {retrieval_time:.2f}s")
                    context_text = generator._format_context(context_docs)
                else:
                    st.warning("⚠️ No direct documents matched your query. Answering based on general knowledge.")
                    context_text = "No direct document matches found."

                # 2. Generation Phase
                final_prompt = generator.prompt_template.format(
                    context=context_text,
                    question=prompt
                )
                
                start_gen = time.perf_counter()
                stream_started = False
                
                for chunk in generator.llm.stream(final_prompt):
                    if not stream_started: stream_started = True
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                end_gen = time.perf_counter()
                message_placeholder.markdown(full_response)
                
                # 3. Guard & Response Metrics
                if not full_response.strip():
                    st.error("The AI engine failed to provide a response. Check service logs.")
                    full_response = "I couldn't generate a response. Please verify the AI connection."
                    message_placeholder.markdown(full_response)
                
                duration = end_gen - start_gen if stream_started else 0
                tps = len(full_response.split()) / duration if duration > 0 else 0
                
                # Fetch detailed retrieval metrics
                hybrid_time = getattr(generator.retriever, "last_hybrid_time", 0.0)
                rerank_time = getattr(generator.retriever, "last_rerank_time", 0.0)
                
                st.caption(
                    f"🚀 Speed: {tps:.2f} tokens/s | Latency: {duration:.2f}s | "
                    f"Retrieval (Total): {retrieval_time:.2f}s "
                    f"(Hybrid: {hybrid_time:.2f}s, Rerank: {rerank_time:.2f}s)"
                )

                # 4. Source Citations
                unique_sources = []
                seen_links = set()
                for d in context_docs:
                    link = d.metadata.get("link")
                    if link and link not in seen_links:
                        unique_sources.append({
                            "filename": Path(d.metadata.get("filename", "Unknown")).name,
                            "link": link,
                            "page": d.metadata.get("page", "?")
                        })
                        seen_links.add(link)
                
                if unique_sources:
                    with st.expander("📌 Source References"):
                        for source in unique_sources:
                            st.markdown(f"- **[{source['filename']}]({source['link']})** • Page {source['page']}")
                
                # 5. history Update
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": unique_sources
                })

            except Exception as e:
                st.error(f"System Error: {e}")
        else:
            st.error("Intelligence engine is offline. Please check backend services.")
