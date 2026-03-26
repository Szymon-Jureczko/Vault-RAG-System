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
    page_icon="🔒",
    layout="wide",
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

    if st.button("Sync Documents", type="primary"):
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
                    event_type = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()

                if event_type == "citations":
                    results_data = json.loads(data_str)

                elif event_type == "token":
                    token = json.loads(data_str)
                    answer_tokens.append(token)
                    answer_placeholder.markdown("".join(answer_tokens))

                elif event_type == "done":
                    break

            answer = "".join(answer_tokens) or "No answer generated."
            answer_placeholder.markdown(answer)

            # Show citations below the completed answer
            if results_data:
                with st.expander("Citations & Sources"):
                    for r in results_data:
                        c = r["citation"]
                        page_str = (
                            f", Page {c['page']}"
                            if c.get("page")
                            else ""
                        )
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
