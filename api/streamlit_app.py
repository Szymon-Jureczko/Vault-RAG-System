"""Streamlit Researcher Dashboard — chat, citations, ingestion.

Run with::

    streamlit run api/streamlit_app.py --server.port 8888
"""

from __future__ import annotations

import json
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LocalVaultRAG — Researcher Dashboard",
    layout="wide",
)

# ── Custom CSS: Black & White, minimal chrome ────────────────────────────────

st.markdown(
    """
<style>
/* ── Import a clean monospace font ─── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');

/* ── Root overrides ───────────────────────────────────────────── */
:root {
    --bg:       #0a0a0a;
    --surface:  #111111;
    --border:   #252525;
    --text:     #d4d4d4;
    --muted:    #6b6b6b;
    --accent:   #ffffff;
}

/* ── Global ───────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* hide streamlit footer & decoration (keep header for sidebar toggle) */
footer, [data-testid="stDecoration"] {
    display: none !important;
}

/* style the header bar to blend in */
header[data-testid="stHeader"] {
    background-color: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}


/* ── Chat avatars — hide entirely ─── */
[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}
[data-testid="stChatMessage"] [data-testid*="chatAvatar"],
[data-testid="stChatMessage"] [data-testid*="Avatar"],
[data-testid="stChatMessage"] .stChatMessageAvatarContainer,
[data-testid="stChatMessage"] img[data-testid*="avatar"],
[data-testid="stChatMessage"] [class*="avatar"],
[data-testid="stChatMessage"] [class*="Avatar"] {
    display: none !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Headings ─────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 1.5rem !important; }

/* ── Paragraphs, captions, labels ─── */
p, label, .stMarkdown,
[data-testid="stCaptionContainer"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stButton > button:hover {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
    border-color: var(--accent) !important;
}

/* primary button — black bg, white text */
button[data-testid="stBaseButton-primary"],
button[kind="primary"],
.stButton button[kind="primary"],
div[data-testid="stBaseButton-primary"],
.stButton [data-baseweb="button"][kind="primary"] {
    background-color: var(--bg) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    font-weight: 500 !important;
}

button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── Text inputs ──────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* ── Radio buttons ────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-size: 0.82rem !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: var(--text) !important;
}

/* ── Chat messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

/* user vs assistant subtle distinction */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 2px solid var(--accent) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid var(--muted) !important;
}

/* chat avatar */
[data-testid="stChatMessage"] [data-testid*="chatAvatarIcon"] {
    background-color: var(--border) !important;
}

/* chat input */
[data-testid="stChatInput"],
[data-testid="stChatInput"] textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* ── Expander (Citations) ─────────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.streamlit-expanderContent {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 2px 2px !important;
}

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}

/* ── Progress bar ─────────────────────────────────────────────── */
.stProgress > div > div {
    background-color: var(--border) !important;
    border-radius: 0 !important;
}

.stProgress > div > div > div {
    background-color: var(--accent) !important;
    border-radius: 0 !important;
}

/* ── Dividers ─────────────────────────────────────────────────── */
hr, [data-testid="stDivider"] {
    border-color: var(--border) !important;
}

/* ── Alerts (success, error, warning) ─────────────────────────── */
.stAlert {
    background-color: var(--surface) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stAlert"] > div {
    background-color: var(--surface) !important;
}

/* success */
div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {
    border-left: 2px solid var(--accent) !important;
}

/* error */
div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {
    border-left: 2px solid #888 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Misc cleanup ─────────────────────────────────────────────── */
.block-container { padding-top: 2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar: Ingestion Controls ─────────────────────────────────────────────

with st.sidebar:
    st.header("Ingestion Controls")

    source_type = st.radio(
        "Ingestion source",
        ["Local Filesystem", "Azure Blob Storage"],
        horizontal=True,
    )

    if source_type == "Local Filesystem":
        source_dir = st.text_input("Source directory", value="data/")
        azure_conn_str = ""
        azure_container = ""
    else:
        source_dir = "data/"
        azure_conn_str = st.text_input(
            "Connection string",
            type="password",
            placeholder=(
                "DefaultEndpointsProtocol=https;AccountName=...;"
                "AccountKey=...;EndpointSuffix=core.windows.net"
            ),
        )
        azure_container = st.text_input(
            "Container name",
            placeholder="my-documents-container",
        )

    st.markdown(
        "<style>"
        "#sync-btn button {"
        "  background-color: #0a0a0a !important;"
        "  color: #ffffff !important;"
        "  border: 1px solid #ffffff !important;"
        "}"
        "#sync-btn button:hover {"
        "  background-color: #ffffff !important;"
        "  color: #0a0a0a !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="sync-btn"):
        sync_clicked = st.button("Sync Documents")
    if sync_clicked:
        if source_type == "Azure Blob Storage" and (
            not azure_conn_str or not azure_container
        ):
            st.error(
                "Connection string and container name are required "
                "for Azure ingestion."
            )
        else:
            status_text = st.empty()
            progress_bar = st.empty()
            try:
                # Build payload — include Azure fields when relevant
                payload: dict = {
                    "source_dir": source_dir,
                    "ingestion_source": (
                        "AZURE" if source_type == "Azure Blob Storage" else "LOCAL"
                    ),
                }
                if source_type == "Azure Blob Storage":
                    payload["azure_storage_connection_string"] = azure_conn_str
                    payload["azure_container_name"] = azure_container

                # Kick off background job
                resp = requests.post(
                    f"{API_BASE}/sync",
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                job = resp.json()
                job_id = job["job_id"]

                # Poll until complete
                while True:
                    time.sleep(3)
                    poll = requests.get(
                        f"{API_BASE}/sync/{job_id}",
                        timeout=120,
                    )
                    poll.raise_for_status()
                    data = poll.json()

                    # Show live progress while running
                    s = data.get("stats")
                    if s and data["status"] == "running":
                        done = s.get("processed", 0) + s.get("failed", 0)
                        total = s.get("files_total", 0)
                        phase = s.get("phase", "")
                        cur = s.get("current_file", "")
                        if total > 0:
                            progress_bar.progress(
                                min(done / total, 1.0),
                                text=f"{done}/{total} files",
                            )
                        status_text.text(f"Phase: {phase} | Processing: {cur}")

                    if not data["status"].startswith("running"):
                        break

                status_text.empty()
                progress_bar.empty()

                if data["status"] != "completed":
                    st.error(f"Sync failed: {data['status']}")
                else:
                    stats = data["stats"]
                    st.success("Sync complete")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Processed", stats["processed"])
                    col2.metric("Skipped", stats["skipped_unchanged"])
                    col3.metric("Failed", stats["failed"])

                    st.metric("Total Chunks", stats["total_chunks"])

                    total = stats["total_discovered"]
                    if total > 0:
                        done = stats["processed"] + stats["skipped_unchanged"]
                        st.progress(
                            min(done / total, 1.0),
                            text=f"{done}/{total} files complete",
                        )
            except requests.ConnectionError:
                st.error("Cannot connect to API server. " "Is it running on port 8000?")
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.divider()

    # Health check
    if st.button("Check Health"):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=10)
            health = resp.json()
            if health.get("status") == "healthy":
                st.success(f"Healthy — {health['documents']}" " documents indexed")
            else:
                st.warning(f"Degraded: {health.get('error', 'unknown')}")
        except requests.ConnectionError:
            st.error("API server not reachable")

# ── Main: Chat Interface ───────────────────────────────────────────────────

st.title("LocalVaultRAG Researcher")
st.caption("Privacy-first RAG — 100% local inference ")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        tag = "user:" if msg["role"] == "user" else "RAG:"
        st.markdown(f"**{tag}** {msg['content']}")
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    page_str = f", Page {c['page']}" if c.get("page") else ""
                    st.markdown(f"**{c['filename']}**{page_str}")
                    st.text(c["snippet"])
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**user:** {question}")

    # Query API — streaming
    with st.chat_message("assistant"):
        try:
            # Open SSE stream — citations arrive first, then tokens
            resp = requests.post(
                f"{API_BASE}/query/stream",
                json={"question": question, "top_k": 10},
                timeout=200,
                stream=True,
            )
            resp.raise_for_status()

            citations = []
            results_data = []
            answer_placeholder = st.empty()
            answer_tokens: list[str] = []

            event_type = ""
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()

                if event_type == "citations":
                    results_data = json.loads(data_str)

                elif event_type == "token":
                    token = json.loads(data_str)
                    answer_tokens.append(token)
                    answer_placeholder.markdown(
                        "**RAG:** " + "".join(answer_tokens)
                    )

                elif event_type == "done":
                    break

            answer = "".join(answer_tokens) or "No answer generated."
            answer_placeholder.markdown(f"**RAG:** {answer}")

            # Show citations below the completed answer
            if results_data:
                with st.expander("Citations & Sources"):
                    for r in results_data:
                        c = r["citation"]
                        page_str = f", Page {c['page']}" if c.get("page") else ""
                        st.markdown(
                            f"**{c['filename']}**"
                            f"{page_str} "
                            f"(score: {r['score']:.3f})"
                        )
                        st.text(c["snippet"])
                        st.divider()
                        citations.append(c)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )

        except requests.ConnectionError:
            err = (
                "Cannot connect to API server. " "Start it with: `uvicorn api.main:app`"
            )
            st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                }
            )
        except Exception as e:
            err = f"Query failed: {e}"
            st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                }
            )
