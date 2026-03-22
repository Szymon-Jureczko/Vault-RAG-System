"""Tests for Azure Blob Storage ingestion source.

Covers Settings Azure fields, the download_azure_blobs function,
and the pipeline's Azure dispatch logic in run().
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.pipeline import (
    SUPPORTED_EXTENSIONS,
    IngestionPipeline,
    download_azure_blobs,
)

# ── Mock helpers ──────────────────────────────────────────────────────────────


def _make_blob(name: str) -> MagicMock:
    """Return a mock blob descriptor with a .name attribute."""
    b = MagicMock()
    b.name = name
    return b


def _build_azure_modules(
    blobs: list[MagicMock],
    download_content: bytes = b"mock file content",
    connect_error: Exception | None = None,
    list_error: Exception | None = None,
    per_blob_errors: dict[str, Exception] | None = None,
) -> MagicMock:
    """Build a mock azure.storage.blob module hierarchy.

    Args:
        blobs: Blob descriptors returned by list_blobs().
        download_content: Bytes written to the file handle for happy-path blobs.
        connect_error: If set, from_connection_string raises this exception.
        list_error: If set, list_blobs raises this exception.
        per_blob_errors: Map of blob name -> exception for per-blob failures.

    Returns:
        Mock module to inject into sys.modules as ``azure.storage.blob``.
    """
    per_blob_errors = per_blob_errors or {}

    mock_module = MagicMock()
    mock_bsc_class = MagicMock()
    mock_service = MagicMock()
    mock_container = MagicMock()

    mock_module.BlobServiceClient = mock_bsc_class

    if connect_error:
        mock_bsc_class.from_connection_string.side_effect = connect_error
        return mock_module

    mock_bsc_class.from_connection_string.return_value = mock_service
    mock_service.get_container_client.return_value = mock_container

    if list_error:
        mock_container.list_blobs.side_effect = list_error
        return mock_module

    mock_container.list_blobs.return_value = blobs

    def _get_blob_client(name: str) -> MagicMock:
        bc = MagicMock()
        if name in per_blob_errors:
            bc.download_blob.side_effect = per_blob_errors[name]
        else:
            stream = MagicMock()
            stream.readinto.side_effect = lambda fh: fh.write(download_content)
            bc.download_blob.return_value = stream
        return bc

    mock_container.get_blob_client.side_effect = _get_blob_client

    return mock_module


def _azure_sys_modules(mock_module: MagicMock) -> dict:
    """Return a sys.modules patch dict for the azure SDK."""
    return {
        "azure": MagicMock(),
        "azure.storage": MagicMock(),
        "azure.storage.blob": mock_module,
    }


# ── Settings: Azure fields ────────────────────────────────────────────────────


class TestSettingsAzureFields:
    """Verify the three new Settings fields and the case-normalisation validator."""

    def test_defaults(self) -> None:
        """ingestion_source defaults to LOCAL; Azure fields default to empty."""
        from ingestion.config import Settings

        s = Settings(_env_file=None)
        assert s.ingestion_source == "LOCAL"
        assert s.azure_storage_connection_string == ""
        assert s.azure_container_name == ""

    def test_uppercase_value_accepted(self) -> None:
        """AZURE (uppercase) is a valid ingestion_source value."""
        from ingestion.config import Settings

        s = Settings(_env_file=None, ingestion_source="AZURE")
        assert s.ingestion_source == "AZURE"

    def test_lowercase_normalised_to_uppercase(self) -> None:
        """'azure' (lowercase) is up-cased to 'AZURE' by the validator."""
        from ingestion.config import Settings

        s = Settings(_env_file=None, ingestion_source="azure")
        assert s.ingestion_source == "AZURE"

    def test_mixed_case_normalised(self) -> None:
        """'Azure' (mixed case) is normalised to 'AZURE'."""
        from ingestion.config import Settings

        s = Settings(_env_file=None, ingestion_source="Azure")
        assert s.ingestion_source == "AZURE"

    def test_invalid_source_raises_validation_error(self) -> None:
        """An unsupported value like 'S3' is rejected by the Literal type."""
        from pydantic import ValidationError

        from ingestion.config import Settings

        with pytest.raises(ValidationError):
            Settings(_env_file=None, ingestion_source="S3")


# ── download_azure_blobs ──────────────────────────────────────────────────────


class TestDownloadAzureBlobs:
    """Tests for the download_azure_blobs module-level function."""

    def test_downloads_supported_blobs_and_returns_count(self, tmp_path: Path) -> None:
        """Supported blobs are written to dest; count equals successful downloads."""
        blobs = [_make_blob("report.pdf"), _make_blob("notes.docx")]
        mock_module = _build_azure_modules(blobs)

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            count = download_azure_blobs(tmp_path)

        assert count == 2
        assert (tmp_path / "report.pdf").exists()
        assert (tmp_path / "notes.docx").exists()

    def test_skips_unsupported_extension(self, tmp_path: Path) -> None:
        """Blobs with extensions not in SUPPORTED_EXTENSIONS are silently skipped."""
        blobs = [_make_blob("config.yaml"), _make_blob("readme.pdf")]
        mock_module = _build_azure_modules(blobs)

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            count = download_azure_blobs(tmp_path)

        assert not (tmp_path / "config.yaml").exists()
        assert count == 1  # only readme.pdf (a supported extension)

    def test_preserves_virtual_directory_structure(self, tmp_path: Path) -> None:
        """Blob virtual paths are mirrored as subdirectory trees under dest."""
        blobs = [_make_blob("finance/q1/summary.pdf")]
        mock_module = _build_azure_modules(blobs)

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            download_azure_blobs(tmp_path)

        assert (tmp_path / "finance" / "q1" / "summary.pdf").exists()

    def test_connection_error_raises_runtime_error(self, tmp_path: Path) -> None:
        """Exception from from_connection_string is wrapped in RuntimeError."""
        mock_module = _build_azure_modules([], connect_error=Exception("Auth failed"))

        with (
            pytest.raises(RuntimeError, match="Failed to initialise"),
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "bad-conn"
            mock_settings.azure_container_name = "docs"
            download_azure_blobs(tmp_path)

    def test_list_blobs_error_raises_runtime_error(self, tmp_path: Path) -> None:
        """Exception from list_blobs is wrapped in RuntimeError."""
        mock_module = _build_azure_modules(
            [], list_error=Exception("Container not found")
        )

        with (
            pytest.raises(RuntimeError, match="Failed to list blobs"),
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "missing"
            download_azure_blobs(tmp_path)

    def test_per_blob_error_is_warned_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failing individual blob is warned and skipped; others still succeed."""
        import logging

        blobs = [_make_blob("good.pdf"), _make_blob("bad.pdf")]
        mock_module = _build_azure_modules(
            blobs, per_blob_errors={"bad.pdf": Exception("network timeout")}
        )

        with (
            caplog.at_level(logging.WARNING, logger="ingestion.pipeline"),
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            count = download_azure_blobs(tmp_path)

        assert count == 1
        assert (tmp_path / "good.pdf").exists()
        assert not (tmp_path / "bad.pdf").exists()
        assert any("bad.pdf" in r.message for r in caplog.records)

    def test_missing_azure_sdk_raises_import_error(self, tmp_path: Path) -> None:
        """ImportError with a helpful pip message when azure-storage-blob is absent."""
        with (
            pytest.raises(ImportError, match="azure-storage-blob"),
            patch.dict(sys.modules, {"azure.storage.blob": None}),
        ):
            download_azure_blobs(tmp_path)

    def test_empty_container_returns_zero(self, tmp_path: Path) -> None:
        """An empty container downloads nothing and returns 0."""
        mock_module = _build_azure_modules([])

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch.dict(sys.modules, _azure_sys_modules(mock_module)),
        ):
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "empty"
            count = download_azure_blobs(tmp_path)

        assert count == 0
        assert list(tmp_path.iterdir()) == []


# ── IngestionPipeline: Azure dispatch in run() ────────────────────────────────


class TestIngestionPipelineAzureSource:
    """Verify run() correctly dispatches to Azure download and manages temp dirs."""

    @pytest.fixture()
    def mock_tracker(self) -> MagicMock:
        """Provide a mock StateTracker that skips SQLite I/O."""
        tracker = MagicMock()
        tracker.purge_deleted.return_value = []
        tracker.needs_processing.return_value = False
        return tracker

    def _make_pipeline(
        self, mock_tracker: MagicMock, tmp_path: Path
    ) -> IngestionPipeline:
        """Construct a pipeline with injected tracker and temp chroma path."""
        return IngestionPipeline(
            tracker=mock_tracker,
            chroma_path=tmp_path / "chroma",
            max_workers=1,
            ocr_workers=1,
        )

    def test_local_source_does_not_call_download(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """In LOCAL mode, download_azure_blobs is never invoked."""
        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.download_azure_blobs") as mock_dl,
            patch("ingestion.pipeline.discover_files", return_value=[]),
        ):
            mock_settings.ingestion_source = "LOCAL"
            pipeline = self._make_pipeline(mock_tracker, tmp_path)
            pipeline.run(tmp_path)

        mock_dl.assert_not_called()

    def test_azure_source_calls_download_before_discover(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """In AZURE mode, download_azure_blobs is called before discover_files."""
        call_order: list[str] = []

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch(
                "ingestion.pipeline.download_azure_blobs",
                side_effect=lambda _: call_order.append("download"),
            ),
            patch(
                "ingestion.pipeline.discover_files",
                side_effect=lambda _: call_order.append("discover") or [],
            ),
        ):
            mock_settings.ingestion_source = "AZURE"
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            pipeline = self._make_pipeline(mock_tracker, tmp_path)
            pipeline.run(tmp_path)

        assert call_order == ["download", "discover"]

    def test_azure_raises_on_missing_connection_string(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """ValueError is raised before any download if conn string is absent."""
        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.download_azure_blobs") as mock_dl,
        ):
            mock_settings.ingestion_source = "AZURE"
            mock_settings.azure_storage_connection_string = ""
            mock_settings.azure_container_name = "docs"
            pipeline = self._make_pipeline(mock_tracker, tmp_path)

            with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
                pipeline.run(tmp_path)

        mock_dl.assert_not_called()

    def test_azure_raises_on_missing_container_name(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """ValueError is raised before any download if container name is absent."""
        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.download_azure_blobs") as mock_dl,
        ):
            mock_settings.ingestion_source = "AZURE"
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = ""
            pipeline = self._make_pipeline(mock_tracker, tmp_path)

            with pytest.raises(ValueError, match="AZURE_CONTAINER_NAME"):
                pipeline.run(tmp_path)

        mock_dl.assert_not_called()

    def test_temp_dir_cleaned_up_after_successful_run(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """The Azure temporary directory is removed after a successful run."""
        created_dirs: list[Path] = []
        _orig = tempfile.TemporaryDirectory

        def _capturing(**kwargs):
            td = _orig(**kwargs)
            created_dirs.append(Path(td.name))
            return td

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch("ingestion.pipeline.download_azure_blobs"),
            patch("ingestion.pipeline.discover_files", return_value=[]),
            patch(
                "ingestion.pipeline.tempfile.TemporaryDirectory",
                side_effect=_capturing,
            ),
        ):
            mock_settings.ingestion_source = "AZURE"
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            pipeline = self._make_pipeline(mock_tracker, tmp_path)
            pipeline.run(tmp_path)

        assert len(created_dirs) == 1
        assert not created_dirs[0].exists()

    def test_temp_dir_cleaned_up_on_download_error(
        self, tmp_path: Path, mock_tracker: MagicMock
    ) -> None:
        """The Azure temp dir is removed even when download_azure_blobs raises."""
        created_dirs: list[Path] = []
        _orig = tempfile.TemporaryDirectory

        def _capturing(**kwargs):
            td = _orig(**kwargs)
            created_dirs.append(Path(td.name))
            return td

        with (
            patch("ingestion.pipeline.settings") as mock_settings,
            patch(
                "ingestion.pipeline.download_azure_blobs",
                side_effect=RuntimeError("network unreachable"),
            ),
            patch(
                "ingestion.pipeline.tempfile.TemporaryDirectory",
                side_effect=_capturing,
            ),
        ):
            mock_settings.ingestion_source = "AZURE"
            mock_settings.azure_storage_connection_string = "conn"
            mock_settings.azure_container_name = "docs"
            pipeline = self._make_pipeline(mock_tracker, tmp_path)

            with pytest.raises(RuntimeError, match="network unreachable"):
                pipeline.run(tmp_path)

        assert len(created_dirs) == 1
        assert not created_dirs[0].exists()
