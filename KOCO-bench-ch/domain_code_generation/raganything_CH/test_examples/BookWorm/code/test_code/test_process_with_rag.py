import asyncio
import sys
import os
import logging
import threading
import time
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from mock_server import MockOpenAIServer

import pytest

# --- Path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from raganything_example import process_with_rag, configure_logging

logger = logging.getLogger("lightrag")


# ==============================================================================
# 1. Pytest Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def mock_openai_server():
    """Starts a mock OpenAI server for integration tests."""
    server = MockOpenAIServer(port=8008)
    server.start()
    yield f"http://localhost:{server.port}/v1"
    server.stop()


# ==============================================================================
# 2. UNIT TESTS (no network, no server, pure mocking)
# ==============================================================================

@pytest.mark.asyncio
async def test_process_with_rag_unit():
    """
    Unit test: mocks RAGAnything instance completely.
    No HTTP calls, no file I/O (mocked).
    """
    mock_rag_instance = MagicMock()
    mock_rag_instance.process_document_complete = AsyncMock(return_value=None)
    mock_rag_instance.aquery = AsyncMock(return_value="mocked answer")
    mock_rag_instance.aquery_with_multimodal = AsyncMock(return_value="mocked multimodal answer")

    with patch("raganything_example.RAGAnything", return_value=mock_rag_instance):
        with patch("builtins.open", mock_open(read_data="fake file content")):
            result = await process_with_rag(
                file_path="fake.txt",
                output_dir="out",
                api_key="fake_key",
                base_url="https://api.openai.com/v1"  # doesn't matter in unit test
            )
            # Verify interactions
            mock_rag_instance.process_document_complete.assert_awaited_once()
            assert mock_rag_instance.aquery.await_count == 2
            assert mock_rag_instance.aquery_with_multimodal.await_count == 2
    print("✅ Unit test passed!")


# ==============================================================================
# 3. INTEGRATION TESTS (uses real HTTP to mock server)
# ==============================================================================

@pytest.mark.asyncio
async def test_rag_integration_with_mock_server(mock_openai_server):
    """
    Integration test: uses real RAGAnything + real HTTP calls to mock server.
    No mocking of RAG internals.
    """
    configure_logging()
    test_file = os.path.join(current_dir, "testfile", "dummy_file.pdf")
    output_dir = os.path.join(current_dir, "mock_output_integration")

    # Ensure test file exists
    assert os.path.exists(test_file), f"Test file not found: {test_file}"

    await process_with_rag(
        file_path=test_file,
        output_dir=output_dir,
        api_key="MOCK_KEY",
        base_url=mock_openai_server,  # e.g., "http://localhost:8008/v1"
    )
    print("✅ Integration test passed!")


# ==============================================================================
# 4. Manual Run Support (optional)
# ==============================================================================

async def run_integration_test_manually():
    server = MockOpenAIServer(port=8008)
    server.start()
    try:
        await process_with_rag(
            file_path="./testfile/dummy_file.pdf",
            output_dir="./mock_output_manual",
            api_key="MOCK_KEY",
            base_url="http://localhost:8008/v1",
        )
        print("✅ Manual integration test finished!")
    finally:
        server.stop()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run_integration_test_manually())