"""Streamlit Researcher Dashboard — chat, citations, ingestion.

Run with::

    streamlit run api/streamlit_app.py --server.port 8888
"""

from __future__ import annotations

import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LocalVaultRAG — Researcher Dashboard",
    page_icon="🔒",
    layout="wide",
)

# ── Sidebar: Ingestion Controls ─────────────────────────────────────────────

with st.sidebar:
    st.header("Ingestion Controls")
    source_dir = st.text_input("Source directory", value="data/")

    if st.button("Sync Documents", type="primary"):
        with st.spinner("Running incremental ingestion..."):
            try:
                # Kick off background job
                resp = requests.post(
                    f"{API_BASE}/sync",
                    json={"source_dir": source_dir},
                    timeout=30,
                )
                resp.raise_for_status()
                job = resp.json()
                job_id = job["job_id"]

                # Poll until complete
                while True:
                    time.sleep(3)
                    poll = requests.get(f"{API_BASE}/sync/{job_id}", timeout=10)
                    poll.raise_for_status()
                    data = poll.json()
                    if not data["status"].startswith("running"):
                        break

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
st.caption("Privacy-first RAG — 100% local inference, zero data leakage")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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
        st.markdown(question)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "top_k": 5},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer generated.")
                st.markdown(answer)

                citations = []
                if data.get("results"):
                    with st.expander("Citations & Sources"):
                        for r in data["results"]:
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
                    "Cannot connect to API server. "
                    "Start it with: `uvicorn api.main:app`"
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
