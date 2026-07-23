# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for **all** dependency management and running the server. **Never invoke `pip` or bare `python`/`uvicorn` directly** — it would bypass `uv.lock` and desync the environment.

```bash
# Install dependencies
uv sync

# Add/remove a dependency (updates pyproject.toml + uv.lock)
uv add <package>
uv remove <package>

# Run the app (from repo root) — Windows requires Git Bash
./run.sh

# Run manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App serves at `http://localhost:8000` (web UI) and `http://localhost:8000/docs` (FastAPI/Swagger docs).

Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

There are no linting, formatting, or test commands/configs in this repo currently — don't assume `pytest`, `ruff`, etc. are set up unless you add them yourself.

## Architecture

This is a Retrieval-Augmented Generation (RAG) chatbot: FastAPI backend (`backend/`) + static HTML/JS frontend (`frontend/`), using ChromaDB for vector storage and Anthropic's Claude for generation. Course transcripts in `docs/` are auto-ingested into ChromaDB on server startup (`app.py`'s `startup_event`), skipping courses already present.

### Request flow (query lifecycle)

`frontend/script.js` POSTs to `/api/query` → `app.py` creates a session if needed → `RAGSystem.query()` (`rag_system.py`) is the orchestrator:

1. Builds a system prompt + retrieves session history from `SessionManager` (in-memory, capped at `MAX_HISTORY` exchanges — lost on restart).
2. Calls `AIGenerator.generate_response()` (`ai_generator.py`), passing the `search_course_content` tool definition.
3. Claude decides whether to answer from general knowledge directly, or invoke the tool for course-specific questions.
4. If tool use is requested (`stop_reason == "tool_use"`), `AIGenerator._handle_tool_execution` runs the tool via `ToolManager` → `CourseSearchTool.execute()` (`search_tools.py`), then makes a **second** Claude call (no tools this round) with the tool results appended, to produce the final synthesized answer.
5. **Only one tool-execution round-trip is supported** — there is no multi-step/looping tool-use agent here.
6. Sources found during search are tracked on the tool instance (`CourseSearchTool.last_sources`) and pulled/reset by `RAGSystem.query()` after each call — do not assume they persist across requests.

### Vector store design (`vector_store.py`)

`VectorStore` wraps two separate ChromaDB collections:
- `course_catalog` — one entry per course (title, instructor, lessons-as-JSON), used purely for **semantic course-name resolution** (e.g. resolving a fuzzy name like "MCP" to the exact stored course title) before filtering.
- `course_content` — the actual chunked text, embedded with `all-MiniLM-L6-v2`, filtered by exact `course_title`/`lesson_number` metadata after resolution.

This two-step resolve-then-filter design is why `CourseSearchTool` accepts a loose `course_name` string rather than requiring an exact title.

### Document processing (`document_processor.py`)

Course `.txt` files have a fixed header format (`Course Title:`, `Course Link:`, `Course Instructor:` on the first lines) followed by `Lesson N: <title>` markers. Chunking is sentence-aware with configurable overlap (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`); the first chunk of each lesson gets a `"Lesson N content: ..."` prefix injected for retrieval context.

### Config (`backend/config.py`)

Single `Config` dataclass loaded via `.env` — model name, chunk size/overlap, `MAX_RESULTS`, `MAX_HISTORY`, and the ChromaDB path (`./chroma_db`, created relative to wherever the process is run from — i.e. `backend/chroma_db` in normal usage). Change model/embedding/chunking behavior here, not inline in other modules.
