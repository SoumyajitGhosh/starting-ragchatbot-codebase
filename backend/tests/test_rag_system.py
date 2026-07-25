"""Integration tests for RAGSystem.query() handling content-related questions.

These run against the real, already-ingested backend/chroma_db and the real
OpenAI API (see conftest.real_config / conftest.rag_system), since the goal
is to observe how the live system actually behaves for content queries, not
just how the pieces behave in isolation.
"""

import pytest

pytestmark = pytest.mark.integration


def test_content_question_returns_answer_and_sources(rag_system):
    answer, sources = rag_system.query(
        "What is covered in lesson 0 of the Building Towards Computer Use with Anthropic course?"
    )

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert len(sources) > 0


def test_general_knowledge_question_skips_tool_use(rag_system):
    answer, sources = rag_system.query("What is 2+2? Answer with just the number.")

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert sources == []


def test_unknown_course_name_handled_gracefully(rag_system):
    # No course by this name exists in docs/ - the system must not raise,
    # regardless of what it ends up answering.
    answer, sources = rag_system.query(
        "What does lesson 1 of the Underwater Basket Weaving course teach?"
    )

    assert isinstance(answer, str)
    assert answer.strip() != ""


def test_lesson_specific_question_returns_matching_lesson_source(rag_system):
    """Regression test: asking about a specific lesson must retrieve content
    from that lesson, not from unrelated lessons in the same course."""
    answer, sources = rag_system.query("What is covered in lesson 2 of the MCP course?")

    assert answer.strip() != ""
    assert any("Lesson 2" in s["text"] for s in sources), (
        f"Expected a Lesson 2 source for a lesson-2-specific question, got: {sources}"
    )


def test_session_history_persists_across_queries(rag_system):
    session_id = rag_system.session_manager.create_session()

    first_answer, _ = rag_system.query(
        "What is the Building Towards Computer Use with Anthropic course about?",
        session_id=session_id,
    )
    assert first_answer.strip() != ""

    second_answer, _ = rag_system.query(
        "Who teaches it?",
        session_id=session_id,
    )

    assert isinstance(second_answer, str)
    assert second_answer.strip() != ""

    history = rag_system.session_manager.get_conversation_history(session_id)
    assert history is not None
    assert "Building Towards Computer Use with Anthropic" in history or "it" in history.lower()
