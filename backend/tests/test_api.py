"""API endpoint tests for the FastAPI app (/api/query, /api/courses, /).

backend/app.py mounts StaticFiles(directory="../frontend") and constructs a
real RAGSystem at import time, so importing it directly here would require
matching its expected cwd and would spin up real ChromaDB/OpenAI clients.
Instead this module builds an equivalent FastAPI app inline - mirroring the
routes defined in app.py - wired to a mocked RAGSystem, so only the
request/response contract of each endpoint is under test.
"""

from typing import List, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import BaseModel

pytestmark = pytest.mark.api


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


class ClearSessionRequest(BaseModel):
    session_id: str


def create_test_app(rag_system, static_dir) -> FastAPI:
    """Build a FastAPI app mirroring app.py's routes, wired to `rag_system`."""
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: ClearSessionRequest):
        try:
            rag_system.session_manager.clear_session(request.session_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Mirrors app.py's static mount, but against a throwaway directory so
    # tests don't depend on the real frontend/ folder existing relative to cwd.
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


@pytest.fixture
def static_dir(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<html><body>RAG Chatbot</body></html>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def test_app(mock_rag_system, static_dir):
    return create_test_app(mock_rag_system, static_dir)


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_creates_session_when_none_provided(self, client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "new-session-id"
        mock_rag_system.query.return_value = ("42", [])

        response = client.post("/api/query", json={"query": "What is 6*7?"})

        assert response.status_code == 200
        body = response.json()
        assert body == {"answer": "42", "sources": [], "session_id": "new-session-id"}
        mock_rag_system.query.assert_called_once_with("What is 6*7?", "new-session-id")

    def test_uses_provided_session_id(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("An answer", [])

        response = client.post(
            "/api/query", json={"query": "Tell me more", "session_id": "existing-session"}
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-session"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more", "existing-session")

    def test_returns_sources_with_text_and_link(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Lesson 1 covers basics.",
            [{"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"}],
        )

        response = client.post("/api/query", json={"query": "What is in lesson 1?"})

        assert response.status_code == 200
        assert response.json()["sources"] == [
            {"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"}
        ]

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "abc"})

        assert response.status_code == 422

    def test_rag_system_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "boom"})

        assert response.status_code == 500
        assert "vector store unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_course_stats(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }

    def test_analytics_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("chroma down")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "chroma down" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/session/clear
# ---------------------------------------------------------------------------

class TestClearSessionEndpoint:
    def test_clears_session(self, client, mock_rag_system):
        response = client.post("/api/session/clear", json={"session_id": "sess-1"})

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("sess-1")

    def test_missing_session_id_returns_422(self, client):
        response = client.post("/api/session/clear", json={})

        assert response.status_code == 422

    def test_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError(
            "no such session"
        )

        response = client.post("/api/session/clear", json={"session_id": "sess-1"})

        assert response.status_code == 500
        assert "no such session" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET / (static frontend)
# ---------------------------------------------------------------------------

class TestStaticRoot:
    def test_serves_index_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "RAG Chatbot" in response.text
