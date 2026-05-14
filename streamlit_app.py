import streamlit as st
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

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

# --- Dynamic User Styling ---
def get_user_theme(user_id: str):
    """Generates a consistent color theme based on user_id hash."""
    import hashlib
    # Generate hash of user_id
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    
    # Base Hue (0-360)
    hue = hash_val % 360
    
    # Theme colors
    # Primary: The user's specific hue
    # Sidebar: Very light version of the hue
    # User Bubble: Light version of the hue
    return {
        "primary": f"hsl({hue}, 70%, 45%)",
        "primary_hover": f"hsl({hue}, 70%, 35%)",
        "sidebar_bg": f"hsl({hue}, 30%, 97%)",
        "sidebar_border": f"hsl({hue}, 30%, 90%)",
        "user_bubble_bg": f"hsl({hue}, 60%, 94%)",
        "user_bubble_border": f"hsl({hue}, 60%, 85%)",
        "user_bubble_text": f"hsl({hue}, 70%, 20%)"
    }

# Default theme for initialization
user_id_for_style = st.session_state.get("user_id", "tamil")
theme = get_user_theme(user_id_for_style)

# --- Clean Light Theme (Dynamic based on User) ---
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: #ffffff;
        color: #1e293b;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {theme['sidebar_bg']} !important;
        border-right: 1px solid {theme['sidebar_border']};
    }}
    [data-testid="stSidebar"] * {{
        color: #1e293b !important;
    }}
    
    /* Headers & Branding */
    h1, h2, h3 {{
        color: {theme['primary']} !important;
        font-weight: 700 !important;
    }}
    
    /* Chat Bubble Styling */
    /* Assistant Bubble (White/Gray) */
    .stChatMessage[data-testid="stChatMessageAssistant"] {{
        background-color: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px;
        color: #1e293b !important;
    }}
    
    /* User Bubble (Dynamic) */
    .stChatMessage[data-testid="stChatMessageUser"] {{
        background-color: {theme['user_bubble_bg']} !important;
        border: 1px solid {theme['user_bubble_border']} !important;
        border-radius: 12px;
        color: {theme['user_bubble_text']} !important;
    }}
    
    /* Buttons - Dynamic */
    .stButton button {{
        background-color: {theme['primary']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: 0.3s ease;
    }}
    .stButton button:hover {{
        background-color: {theme['primary_hover']} !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }}

    /* Links - Dynamic */
    a {{
        color: {theme['primary']} !important;
        text-decoration: none;
        font-weight: 600;
    }}
    a:hover {{
        text-decoration: underline;
    }}

    /* Input & Widgets */
    .stChatInput {{
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }}
    
    /* Status Messages */
    .stStatusWidget {{
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }}

    /* Management Cards */
    .mgmt-card {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }}
    .user-pill {{
        background-color: #f1f5f9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
    }}
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

def sync_history_from_redis():
    """Syncs the Streamlit session messages with the Redis chat history."""
    if st.session_state.get("system_initialized") and st.session_state.get("session_id"):
        # Fetch from Redis
        redis_messages = st.session_state.generator.history_manager.get_history(st.session_state.session_id)
        sync_messages = []
        for msg in redis_messages:
            role = "user" if msg.type == "human" else "assistant"
            sync_messages.append({"role": role, "content": msg.content})
        
        # Only update if different to avoid unnecessary reruns
        if sync_messages != st.session_state.messages:
            st.session_state.messages = sync_messages
            return True
    return False

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

# --- Sidebar: Navigation & Control ---
with st.sidebar:
    st.title("Endee Intelligence")
    
    # 1. Page Navigation
    page = st.sidebar.radio("Navigate", ["🤖 AI Chat", "⚙️ System Management"], index=0)
    st.divider()

    if page == "🤖 AI Chat":
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

    st.subheader("🚀 Session & Cache Settings")
    
    # Display Namespaces for transparency
    with st.expander("🛠️ Active Namespaces", expanded=False):
        st.code(f"Cache: {settings.semantic_cache_prefix}*\nChat:  {settings.redis_chat_history_prefix}*", language="text")

    # 1. User Selection
    if st.session_state.system_initialized:
        discovered_users = st.session_state.generator.history_manager.list_all_users()
        current_user = st.session_state.get("user_id", "tamil")
        
        # Only show existing registered users
        all_users = discovered_users if discovered_users else [current_user]
        if current_user not in all_users:
            all_users = sorted(all_users + [current_user])
            
        default_user_idx = all_users.index(current_user) if current_user in all_users else 0
        user_id = st.sidebar.selectbox("User ID", options=all_users, index=default_user_idx)
        st.session_state.user_id = user_id
        
        # Ensure current user is ALWAYS registered in the background
        if user_id:
            st.session_state.generator.history_manager.register_user(user_id)
    else:
        user_id = st.text_input("User ID", value="tamil")

    # 2. Session Selection
    if st.session_state.system_initialized:
        user_sessions = st.session_state.generator.history_manager.list_user_sessions(user_id)
        
        col_chat_hdr1, col_chat_hdr2 = st.sidebar.columns([4, 1])
        col_chat_hdr1.subheader("💬 Conversations")
        
        # New Chat Button - Generates automatic title
        if col_chat_hdr2.button("➕", help="Start a New Chat"):
            from datetime import datetime
            new_title = datetime.now().strftime("Chat %Y-%m-%d %H:%M:%S")
            st.session_state.conv_id = new_title
            st.rerun()

        current_conv = st.session_state.get("conv_id", "general")
        
        # Ensure current session is in options
        session_options = sorted(list(set(user_sessions + [current_conv]))) if user_sessions else [current_conv]
        default_idx = session_options.index(current_conv) if current_conv in session_options else 0
        
        conv_id = st.sidebar.selectbox("Select Chat", options=session_options, index=default_idx)
        st.session_state.conv_id = conv_id
            
        # Delete & Rename Actions (Renaming still allowed "whenever wanted")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True, help="Permanently delete this chat"):
                st.session_state.generator.history_manager.clear_history(f"{user_id}:{conv_id}")
                st.success(f"Deleted '{conv_id}'")
                st.session_state.conv_id = "general"
                st.rerun()
        with c2:
            if st.button("✏️ Rename", use_container_width=True, help="Change the name of this chat"):
                st.session_state.show_rename_input = not st.session_state.get("show_rename_input", False)
        
        if st.session_state.get("show_rename_input"):
            new_name = st.sidebar.text_input("New Chat Name", value=conv_id, key="rename_input_field")
            if st.sidebar.button("💾 Save New Name", use_container_width=True):
                if new_name and new_name != conv_id:
                    if st.session_state.generator.history_manager.rename_session(user_id, conv_id, new_name):
                        st.session_state.conv_id = new_name
                        st.session_state.show_rename_input = False
                        st.success(f"Renamed to '{new_name}'")
                        st.rerun()
                    else:
                        st.error("Rename failed.")
    else:
        conv_id = st.text_input("Chat ID", value="general")
    
    # Update Session State
    st.session_state.user_id = user_id
    st.session_state.conv_id = conv_id
    new_session_id = f"{user_id}:{conv_id}"
    
    # Check for session change and sync
    if st.session_state.get("session_id") != new_session_id:
        st.session_state.session_id = new_session_id
        if sync_history_from_redis():
            st.rerun()
    
    st.session_state.tenant_id = user_id
    st.session_state.doc_version = st.text_input("Doc Version", value="1.0", help="Version for semantic cache")

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
    with st.expander("🧹 Cache Management", expanded=False):
        st.subheader("Personal Cache")
        if st.button("Clear My Session History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.system_initialized:
                st.session_state.generator.history_manager.clear_history(st.session_state.session_id)
            st.success("Session history cleared.")
            st.rerun()
            
        if st.button("Clear My Semantic Cache", use_container_width=True):
            if st.session_state.system_initialized:
                tenant_id = st.session_state.get("tenant_id", "default")
                st.session_state.generator.semantic_cache.clear_tenant(tenant_id)
                st.success(f"Semantic cache for '{tenant_id}' cleared.")

        st.subheader("Administrative (Global)")
        if st.button("🗑️ Clear ALL Chat Histories", type="primary", use_container_width=True):
            if st.session_state.system_initialized:
                count = st.session_state.generator.history_manager.clear_all_histories()
                st.success(f"Cleared {count} chat sessions.")
        
        if st.button("🗑️ Clear ALL Semantic Caches", type="primary", use_container_width=True):
            if st.session_state.system_initialized:
                count = st.session_state.generator.semantic_cache.clear_all()
                st.success(f"Cleared {count} semantic cache entries.")

        if st.button("🗑️ Clear ALL Embedding Caches", type="primary", use_container_width=True):
            if st.session_state.system_initialized:
                count = st.session_state.generator.base_retriever.embedding_service.clear_all()
                st.success(f"Cleared {count} cached embeddings.")

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

# --- Main UI Routing ---
if page == "🤖 AI Chat":
    st.title("RAG - Chat")
    st.write("Expert guidance based on your engineering and maintenance documentation.")
    
    # Initialize services
    if not st.session_state.system_initialized:
        with st.spinner("Connecting to Engineering Intelligence Engine..."):
            initialize_services()
elif page == "⚙️ System Management":
    st.title("⚙️ System Management")
    st.write("Complete administrative control over users, history, and performance caches.")
    
    if not st.session_state.system_initialized:
        initialize_services()

    # 1. Dashboard Overview Metrics
    m1, m2, m3 = st.columns(3)
    if st.session_state.system_initialized:
        discovered_users = st.session_state.generator.history_manager.list_all_users()
        current_user = st.session_state.get("user_id", "tamil")
        users = sorted(list(set(discovered_users + [current_user])))
        
        m1.metric("Active Users", len(users))
        total_sessions = sum([len(st.session_state.generator.history_manager.list_user_sessions(u)) for u in users])
        m2.metric("Total Chat Sessions", total_sessions)
        m3.metric("Isolation Mode", "Strict User-Based")

    st.divider()

    # 2. Add New User Section
    with st.container():
        st.markdown('<div class="mgmt-card">', unsafe_allow_html=True)
        st.subheader("➕ Onboard New User")
        with st.form("create_user_form_v2", clear_on_submit=True):
            col_f1, col_f2 = st.columns([3, 1])
            new_user_id = col_f1.text_input("New User ID", placeholder="Enter unique identifier (e.g. jdoe_01)")
            submit_user = col_f2.form_submit_button("✨ Create & Login", use_container_width=True)
            
            if submit_user:
                clean_new_id = new_user_id.strip()
                if clean_new_id:
                    st.session_state.generator.history_manager.register_user(clean_new_id)
                    st.session_state.user_id = clean_new_id
                    st.session_state.conv_id = "general"
                    st.success(f"Success! User '{clean_new_id}' is now active.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please enter a valid User ID.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. User & History Table
    st.header("👥 User Directory & Conversations")
    
    if st.session_state.system_initialized:
        if not users:
            st.info("No users found in the system.")
        else:
            for user in users:
                user_theme = get_user_theme(user)
                with st.container():
                    st.markdown(f"""
                        <div style="border-left: 5px solid {user_theme['primary']}; padding: 15px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0; color: {user_theme['primary']} !important;">👤 {user}</h3>
                                <span class="user-pill">Active Profile</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    c_inner1, c_inner2 = st.columns([3, 1])
                    
                    with c_inner1:
                        st.write("**Recent Conversations**")
                        discovered_sessions = st.session_state.generator.history_manager.list_user_sessions(user)
                        
                        # Include current session if viewing current user
                        if user == st.session_state.get("user_id"):
                            sessions = sorted(list(set(discovered_sessions + [st.session_state.get("conv_id", "general")])))
                        else:
                            sessions = discovered_sessions
                        if not sessions:
                            st.caption("No history found for this user.")
                        else:
                            for sess in sessions:
                                sc1, sc2 = st.columns([5, 1])
                                sc1.markdown(f"📄 `{sess}`")
                                if sc2.button("🗑️", key=f"del_sess_page_{user}_{sess}", help="Delete this chat"):
                                    st.session_state.generator.history_manager.clear_history(f"{user}:{sess}")
                                    st.toast(f"Session '{sess}' wiped.")
                                    time.sleep(0.5)
                                    st.rerun()
                    
                    with c_inner2:
                        st.write("**Security**")
                        if st.button(f"🗑️ Delete User {user}", key=f"wipe_u_{user}", type="primary", use_container_width=True):
                            st.session_state.generator.history_manager.delete_user(user)
                            st.session_state.generator.semantic_cache.clear_tenant(user)
                            st.success(f"User '{user}' data purged.")
                            time.sleep(1)
                            st.rerun()
                st.divider()

    # 4. Global Optimization Controls
    st.header("🧠 Engine Optimization")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown('<div class="mgmt-card">', unsafe_allow_html=True)
        st.subheader("Semantic Cache")
        st.write("Clear the shared answer cache to force fresh LLM responses.")
        if st.button("🧹 Clear Answers", key="clear_sem_all", use_container_width=True):
            count = st.session_state.generator.semantic_cache.clear_all()
            st.success(f"Cleared {count} cached answers.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with g_col2:
        st.markdown('<div class="mgmt-card">', unsafe_allow_html=True)
        st.subheader("Embedding Cache")
        st.write("Clear vector cache to force re-analysis of text chunks.")
        if st.button("🧹 Clear Vectors", key="clear_embed_all", use_container_width=True):
            count = st.session_state.generator.base_retriever.embedding_service.clear_all()
            st.success(f"Cleared {count} vectors.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Only show chat UI if on Chat page
if page == "🤖 AI Chat":

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
                    # 0. Fetch Chat History from Redis
                    with st.spinner("Fetching chat history..."):
                        chat_history = generator.history_manager.get_history(st.session_state.session_id)

                    # 0. Semantic Cache Check
                    with st.spinner("Checking semantic cache..."):
                        normalized_query = generator.semantic_cache.normalize_query(prompt)
                        q_dense = generator.base_retriever.embedding_service.get_dense_embedding(normalized_query)
                        
                        cached_hit = generator.semantic_cache.search(
                            query_embedding=q_dense,
                            tenant_id=st.session_state.get("tenant_id", "default"),
                            doc_version=st.session_state.get("doc_version", "1.0")
                        )

                    if cached_hit and cached_hit.get("hit"):
                        st.success(f"🎯 Semantic Cache Hit! (Similarity: {cached_hit['similarity']:.4f})")
                        full_response = cached_hit["response"]
                        message_placeholder.markdown(full_response)
                        
                        st.caption(
                            f"🚀 Speed: FAST (Cached) | Latency: {cached_hit['search_time']:.4f}s | "
                            f"Tenant: {st.session_state.get('tenant_id', 'default')} | Version: {st.session_state.get('doc_version', '1.0')}"
                        )
                        
                        # Save to Redis Chat History
                        generator.history_manager.add_user_message(st.session_state.session_id, prompt)
                        generator.history_manager.add_ai_message(st.session_state.session_id, full_response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": []
                        })
                    else:
                        if cached_hit and cached_hit.get("similarity", 0) > 0:
                            st.info(f"ℹ️ Cache Miss (Best match similarity: {cached_hit['similarity']:.4f}, Threshold: {settings.semantic_cache_threshold})")
                        
                        # 1. Retrieval Phase
                        with st.spinner("Analyzing manuals..."):
                            if chat_history:
                                # Use history-aware retriever
                                context_docs = generator.history_aware_retriever.invoke({
                                    "input": prompt,
                                    "chat_history": chat_history
                                })
                            else:
                                # Fallback to direct retrieval
                                context_docs = generator.base_retriever.invoke(prompt)
                            
                            retrieval_time = getattr(generator.base_retriever, "last_retrieval_time", 0.0)
                        
                        if context_docs:
                            st.toast(f"🔍 Found {len(context_docs)} relevant context points in {retrieval_time:.2f}s")
                            context_text = generator._format_context(context_docs)
                        else:
                            st.warning("⚠️ No direct documents matched your query. Answering based on general knowledge.")
                            context_text = "No direct document matches found."

                        # 2. Generation Phase
                        formatted_history = generator._format_history(chat_history, limit=settings.history_window_size)
                        final_prompt = generator.prompt_template.format(
                            chat_history=formatted_history,
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
                        hybrid_time = getattr(generator.base_retriever, "last_hybrid_time", 0.0)
                        rerank_time = getattr(generator.base_retriever, "last_rerank_time", 0.0)
                        
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
                        
                        # 5. Store in Semantic Cache
                        generator.semantic_cache.store(
                            query=normalized_query,
                            embedding=q_dense,
                            response=full_response,
                            tenant_id=st.session_state.get("tenant_id", "default"),
                            doc_version=st.session_state.get("doc_version", "1.0")
                        )

                        # 6. Save to Redis Chat History
                        generator.history_manager.add_user_message(st.session_state.session_id, prompt)
                        generator.history_manager.add_ai_message(st.session_state.session_id, full_response)
                        
                        # 7. UI State Update
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": unique_sources
                        })

                except Exception as e:
                    st.error(f"System Error: {e}")
            else:
                st.error("Intelligence engine is offline. Please check backend services.")
