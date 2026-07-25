import json
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
import ai_generator as ai_generator_module
from ai_generator import AIGenerator
from config import config as real_app_config
from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# CourseSearchTool fixtures (mocked VectorStore)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """A populated SearchResults spanning two lessons of one course."""
    return SearchResults(
        documents=[
            "Lesson 0 content: Welcome to Building Towards Computer Use with Anthropic.",
            "In this lesson we cover the basics of the Anthropic API.",
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1, "chunk_index": 0},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_course_link.return_value = "https://example.com/course"
    return store


@pytest.fixture
def search_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def outline_tool(mock_vector_store):
    return CourseOutlineTool(mock_vector_store)


# ---------------------------------------------------------------------------
# AIGenerator fixtures (mocked OpenAI client)
# ---------------------------------------------------------------------------

def _make_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_chat_completion(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def make_tool_call():
    return _make_tool_call


@pytest.fixture
def make_chat_completion():
    return _make_chat_completion


@pytest.fixture
def patched_ai_generator(monkeypatch):
    """AIGenerator wired to a MagicMock OpenAI client instead of a real one."""
    mock_client = MagicMock()
    mock_openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr(ai_generator_module, "OpenAI", mock_openai_ctor)

    generator = AIGenerator(api_key="test-key", model="gpt-4o-mini")
    return generator, mock_client


@pytest.fixture
def mock_tool_manager():
    return MagicMock(spec=ToolManager)


# ---------------------------------------------------------------------------
# API fixtures (mocked RAGSystem, no real ChromaDB/OpenAI calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem, wired with sensible defaults.

    Individual tests override .query / .get_course_analytics / session_manager
    return values or side_effects as needed.
    """
    mock = MagicMock(spec=RAGSystem)
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.query.return_value = ("This is a test answer.", [])
    mock.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
    return mock


# ---------------------------------------------------------------------------
# Real, live-system fixtures (real OpenAI API + already-ingested chroma_db)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_config():
    if not real_app_config.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not set in .env; skipping live integration test")

    # backend/chroma_db already has the docs/ courses ingested. Config.CHROMA_PATH
    # is "./chroma_db", which only resolves correctly when cwd == backend/. Pin it
    # to an absolute path so these tests work regardless of pytest's invocation dir.
    backend_dir = Path(__file__).resolve().parent.parent
    return dataclasses.replace(real_app_config, CHROMA_PATH=str(backend_dir / "chroma_db"))


@pytest.fixture(scope="session")
def rag_system(real_config):
    return RAGSystem(real_config)
