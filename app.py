"""
Streamlit UI — Medical RAG Chatbot (Fully Offline)
====================================================
Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_chatbot import MedRAGChatbot

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* User chat bubble */
        .user-bubble {
            background: #1a4f91;
            color: #ffffff;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0 8px 80px;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        /* Bot chat bubble */
        .bot-bubble {
            background: #f0f4f9;
            color: #1a1a1a;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 80px 4px 0;
            font-size: 0.95rem;
            line-height: 1.6;
            border-left: 4px solid #1a4f91;
        }
        /* Not-found bubble */
        .notfound-bubble {
            background: #fff8e1;
            color: #5d4037;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 80px 4px 0;
            font-size: 0.95rem;
            border-left: 4px solid #f9a825;
        }
        /* Source badge */
        .source-badge {
            display: inline-block;
            background: #e3f0ff;
            color: #1a4f91;
            border: 1px solid #90caf9;
            border-radius: 12px;
            padding: 3px 11px;
            font-size: 0.78rem;
            margin: 3px 3px 3px 0;
        }
        /* Stat card */
        .stat-card {
            background: #f7f9fc;
            border-left: 4px solid #1a4f91;
            border-radius: 6px;
            padding: 9px 13px;
            margin: 5px 0;
            font-size: 0.87rem;
        }
        /* Offline badge */
        .offline-badge {
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
            border-radius: 8px;
            padding: 4px 12px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical RAG Chatbot")
    
    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Documents to retrieve (Top-K)",
        min_value=1, max_value=10, value=5,
        help="Number of relevant documents retrieved per query.",
    )
    min_score = st.slider(
        "Minimum similarity score",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Documents below this score are discarded as irrelevant.",
    )

    # st.divider()

    # # Dataset info
    # st.subheader("📊 Dataset Info")
    # st.markdown('<div class="stat-card">📄 <b>16,412</b> medical Q&A pairs</div>', unsafe_allow_html=True)
    # st.markdown('<div class="stat-card">🏷️ <b>5,127</b> unique medical topics</div>', unsafe_allow_html=True)
    # st.markdown('<div class="stat-card">🏛️ NIH · CDC · GARD · GHR · NHLBI · NIDDK</div>', unsafe_allow_html=True)
    # st.markdown('<div class="stat-card">🤖 Embeddings: all-MiniLM-L6-v2 (local)</div>', unsafe_allow_html=True)
    # st.markdown('<div class="stat-card">🔍 Vector search: FAISS (on-disk index)</div>', unsafe_allow_html=True)

    # st.divider()

    # Sample questions
    st.subheader("💡 Try These Questions")
    samples = [
        "What are the symptoms of diabetes?",
        "How is glaucoma treated?",
        "What causes Alzheimer's disease?",
        "What are the risk factors for heart disease?",
        "How is asthma diagnosed?",
        "What is sickle cell disease?",
        "What are the symptoms of tuberculosis?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True, key=f"s_{q}"):
            st.session_state["pending_query"] = q

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state["messages"]  = []
        st.session_state["meta"]      = {}
        st.rerun()

# ── Load chatbot (cached so it doesn't reload on every interaction) ─────────────
@st.cache_resource(show_spinner=False)
def load_chatbot() -> MedRAGChatbot:
    bot = MedRAGChatbot()
    bot.setup()
    return bot


with st.spinner("Initializing offline RAG pipeline..."):
    bot = load_chatbot()

# Update runtime settings without rebuilding index
bot.top_k      = top_k
bot.min_score  = min_score

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # {"role": "user"|"assistant", "content": str, "found": bool}

if "meta" not in st.session_state:
    st.session_state["meta"] = {}       # msg_index → {"sources": [...], "docs": [...]}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 💬 Medical Knowledge Assistant")
st.caption(
    "Answers are generated **strictly** from local MedQuAD medical documents. "
    "No internet connection or external API is used."
)

# ── Welcome ────────────────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔒 **Fully Offline**\nNo data leaves your machine.")
    with col2:
        st.info("📄 **16,412 Q&A pairs**\nFrom NIH, CDC & more.")
    with col3:
        st.info("🔍 **FAISS Search**\nSemantic vector retrieval.")
    st.markdown("---")

# ── Render chat history ────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state["messages"]):

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑‍⚕️ &nbsp;{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    else:
        bubble_class = "bot-bubble" if msg.get("found", True) else "notfound-bubble"
        icon = "🤖" if msg.get("found", True) else "⚠️"
        # Render markdown inside the bubble via st.markdown (not unsafe HTML)
        # Use a container to visually style it
        with st.container():
            st.markdown(
                f'<div class="{bubble_class}">{icon}&nbsp;&nbsp;',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

        # Source badges
        meta = st.session_state["meta"].get(i, {})
        sources = meta.get("sources", [])
        if sources:
            badges = "".join(
                f'<span class="source-badge">📌 {s}</span>' for s in sources[:5]
            )
            st.markdown(
                f"<div style='margin:2px 0 10px 0'>{badges}</div>",
                unsafe_allow_html=True,
            )

        # Retrieved documents expander
        docs = meta.get("docs", [])
        if docs:
            with st.expander(f"📂 Retrieved Documents ({len(docs)} found)", expanded=False):
                for j, doc in enumerate(docs, 1):
                    score_pct = f"{doc.score * 100:.1f}%"
                    st.markdown(
                        f"**#{j}** &nbsp; `{doc.focus_area}` &nbsp;|&nbsp; "
                        f"Source: `{doc.source}` &nbsp;|&nbsp; "
                        f"Similarity: `{score_pct}`"
                    )
                    st.markdown(f"**Q:** {doc.question}")
                    answer_preview = doc.answer[:500] + ("..." if len(doc.answer) > 500 else "")
                    st.markdown(f"**A:** {answer_preview}")
                    if j < len(docs):
                        st.divider()

# ── Chat input ─────────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Type your medical question here...", key="chat_input")
query = pending or user_input

if query:
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching local medical database..."):
        response = bot.chat(query)

    msg_index = len(st.session_state["messages"])
    st.session_state["messages"].append(
        {
            "role":    "assistant",
            "content": response.answer,
            "found":   response.found,
        }
    )
    st.session_state["meta"][msg_index] = {
        "sources": response.sources,
        "docs":    response.retrieved_docs,
    }

    st.rerun()
