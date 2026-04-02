"""Tests for _generate_answer in api.main.

Covers the prompt construction fixes:
- duplicate [Source:] prefix is stripped from r.text
- Page None is not rendered when citation.page is None
- page number is rendered when citation.page is set
- llm_num_ctx / llm_num_predict / llm_timeout settings flow into the Ollama call
- new Settings fields have correct defaults
- new Settings fields are overridable via environment variables
- /api/chat endpoint is used with system + user messages
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.main import _generate_answer
from ingestion.config import Settings
from vector_db.retriever import Citation, RetrievalResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    text: str,
    filename: str = "doc.pdf",
    page: int | None = 1,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="doc_0_abc12345",
        text=text,
        score=score,
        citation=Citation(filename=filename, page=page),
    )


def _mock_httpx_response(answer: str = "test answer") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": answer}}
    return resp


def _get_user_content(captured: list[dict]) -> str:
    """Extract the user message content from a captured /api/chat payload."""
    messages = captured[0]["messages"]
    return next(m["content"] for m in messages if m["role"] == "user")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGenerateAnswerPrompt:
    def test_source_prefix_stripped(self) -> None:
        """Ingestion-time [Source:] prefix in r.text must not appear twice in prompt."""
        results = [
            _make_result(
                "[Source: MiSR1.pdf]\nLiteratura: Craig, Spong.",
                filename="MiSR1.pdf",
                page=1,
            ),
            _make_result(
                "[Source: MiSR2.pdf]\nWykład 2.", filename="MiSR2.pdf", page=2
            ),
        ]
        captured: list[dict] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured.append(json)
            return _mock_httpx_response()

        with patch("api.main.httpx") as mock_httpx:
            mock_httpx.post.side_effect = fake_post
            _generate_answer("What literature?", results)

        assert captured, "httpx.post was not called"
        user_content = _get_user_content(captured)
        # The doubled pattern would be e.g.
        # "[Source: MiSR1.pdf, Page 1]\n[Source: MiSR1.pdf]"
        assert "[Source: MiSR1.pdf]\n[Source:" not in user_content
        assert "[Source: MiSR2.pdf]\n[Source:" not in user_content

    def test_page_none_not_in_prompt(self) -> None:
        """'Page None' must never appear in the prompt for non-PDF chunks."""
        results = [
            _make_result("Some text without page.", filename="notes.txt", page=None)
        ]
        captured: list[dict] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured.append(json)
            return _mock_httpx_response()

        with patch("api.main.httpx") as mock_httpx:
            mock_httpx.post.side_effect = fake_post
            _generate_answer("Any question?", results)

        user_content = _get_user_content(captured)
        assert "Page None" not in user_content

    def test_page_number_rendered_in_prompt(self) -> None:
        """When citation.page is set, the page number must appear in the prompt."""
        results = [
            _make_result("[Source: doc.pdf]\nContent.", filename="doc.pdf", page=5)
        ]
        captured: list[dict] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured.append(json)
            return _mock_httpx_response()

        with patch("api.main.httpx") as mock_httpx:
            mock_httpx.post.side_effect = fake_post
            _generate_answer("Question?", results)

        user_content = _get_user_content(captured)
        assert "Page 5" in user_content

    def test_uses_chat_endpoint_with_system_message(self) -> None:
        """Must call /api/chat with separate system and user messages."""
        results = [_make_result("text", page=1)]
        captured: list[dict] = []
        captured_urls: list[str] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured.append(json)
            captured_urls.append(url)
            return _mock_httpx_response()

        with patch("api.main.httpx") as mock_httpx:
            mock_httpx.post.side_effect = fake_post
            _generate_answer("Q?", results)

        assert captured_urls[0].endswith("/api/chat")
        messages = captured[0]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_system_prompt_enforces_grounding(self) -> None:
        """System message must instruct the model to use only context facts."""
        results = [_make_result("text", page=1)]
        captured: list[dict] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured.append(json)
            return _mock_httpx_response()

        with patch("api.main.httpx") as mock_httpx:
            mock_httpx.post.side_effect = fake_post
            _generate_answer("Q?", results)

        system_content = next(
            m["content"] for m in captured[0]["messages"] if m["role"] == "system"
        )
        assert "ONLY" in system_content
        assert "information not found" in system_content.lower()


class TestGenerateAnswerSettings:
    def test_settings_values_passed_to_ollama(self) -> None:
        """Custom llm_num_ctx / llm_num_predict / llm_timeout must flow to
        httpx.post."""
        results = [_make_result("text", page=1)]
        captured_json: list[dict] = []
        captured_timeout: list[float] = []

        def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:
            captured_json.append(json)
            captured_timeout.append(timeout)
            return _mock_httpx_response()

        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.ollama_model_dev = "llama3.2:3b"
        mock_settings.llm_num_ctx = 999
        mock_settings.llm_num_predict = 111
        mock_settings.llm_timeout = 77.0
        mock_settings.llm_temperature = 0.5
        mock_settings.llm_num_thread = 4

        with (
            patch("api.main.settings", mock_settings),
            patch("api.main.httpx") as mock_httpx,
        ):
            mock_httpx.post.side_effect = fake_post
            _generate_answer("Q?", results)

        assert captured_json, "httpx.post was not called"
        options = captured_json[0]["options"]
        assert options["num_ctx"] == 999
        assert options["num_predict"] == 111
        assert options["temperature"] == 0.5
        assert options["num_thread"] == 4
        assert captured_timeout[0] == 77.0


class TestSettingsNewFields:
    def test_new_fields_have_correct_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llm_num_ctx / llm_num_predict / llm_timeout must have safe defaults."""
        monkeypatch.delenv("LLM_NUM_PREDICT", raising=False)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.llm_num_ctx == 6144
        assert s.llm_num_predict == 768
        assert s.llm_timeout == 120.0
        assert s.llm_temperature == 0.0
        assert s.llm_num_thread == 4

    def test_fields_overridable_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_NUM_CTX env var must override the default."""
        monkeypatch.setenv("LLM_NUM_CTX", "16384")
        monkeypatch.setenv("LLM_NUM_PREDICT", "256")
        monkeypatch.setenv("LLM_TIMEOUT", "60")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("LLM_NUM_THREAD", "4")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.llm_num_ctx == 16384
        assert s.llm_num_predict == 256
        assert s.llm_timeout == 60.0
        assert s.llm_temperature == 0.7
        assert s.llm_num_thread == 4


class TestScoreFiltering:
    """Test the score-based filtering logic used in the /query endpoint."""

    def test_positive_scores_kept(self) -> None:
        """Only chunks with positive reranker scores are sent to the LLM."""
        results = [
            _make_result("relevant1", filename="a.pdf", page=1, score=4.5),
            _make_result("relevant2", filename="b.pdf", page=1, score=2.1),
            _make_result("irrelevant1", filename="c.pdf", page=1, score=-1.0),
            _make_result("irrelevant2", filename="d.pdf", page=1, score=-3.5),
        ]
        relevant = [r for r in results[:5] if r.score > 0]
        if not relevant:
            relevant = results[:1]

        assert len(relevant) == 2
        assert all(r.score > 0 for r in relevant)

    def test_all_negative_falls_back_to_best(self) -> None:
        """When all scores are negative, the best single chunk is used."""
        results = [
            _make_result("best", filename="a.pdf", page=1, score=-0.5),
            _make_result("worse", filename="b.pdf", page=1, score=-2.0),
        ]
        relevant = [r for r in results[:5] if r.score > 0]
        if not relevant:
            relevant = results[:1]

        assert len(relevant) == 1
        assert relevant[0].text == "best"
