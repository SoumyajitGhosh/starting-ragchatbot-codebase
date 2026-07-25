"""Unit tests for CourseSearchTool.execute() with a mocked VectorStore."""

from vector_store import SearchResults


def test_execute_formats_results_with_course_and_lesson_headers(search_tool, mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results

    result = search_tool.execute(query="what is covered")

    assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
    assert "[Building Towards Computer Use with Anthropic - Lesson 1]" in result
    assert "Welcome to Building Towards Computer Use with Anthropic." in result
    assert "we cover the basics of the Anthropic API" in result


def test_execute_tracks_sources_with_links(search_tool, mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results

    search_tool.execute(query="what is covered")

    assert search_tool.last_sources == [
        {"text": "Building Towards Computer Use with Anthropic - Lesson 0", "link": "https://example.com/lesson"},
        {"text": "Building Towards Computer Use with Anthropic - Lesson 1", "link": "https://example.com/lesson"},
    ]
    mock_vector_store.get_lesson_link.assert_any_call("Building Towards Computer Use with Anthropic", 0)
    mock_vector_store.get_lesson_link.assert_any_call("Building Towards Computer Use with Anthropic", 1)


def test_execute_no_results_message_plain(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = search_tool.execute(query="nonexistent topic")

    assert result == "No relevant content found."


def test_execute_no_results_message_includes_filters(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = search_tool.execute(query="nonexistent topic", course_name="MCP", lesson_number=3)

    assert result == "No relevant content found in course 'MCP' in lesson 3."


def test_execute_propagates_store_error(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty("No course found matching 'Nonexistent Course'")

    result = search_tool.execute(query="anything", course_name="Nonexistent Course")

    assert result == "No course found matching 'Nonexistent Course'"


def test_execute_passes_query_course_lesson_to_store(search_tool, mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results

    search_tool.execute(query="find this", course_name="Computer Use", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="find this", course_name="Computer Use", lesson_number=2
    )


def test_execute_lesson_without_link_does_not_crash(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["some content"],
        metadata=[{"course_title": "Some Course", "lesson_number": 5}],
        distances=[0.1],
    )
    mock_vector_store.get_lesson_link.return_value = None

    result = search_tool.execute(query="anything")

    assert "[Some Course - Lesson 5]" in result
    assert search_tool.last_sources == [{"text": "Some Course - Lesson 5", "link": None}]


def test_execute_result_without_lesson_number_uses_course_link(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["course-level content"],
        metadata=[{"course_title": "Some Course"}],
        distances=[0.1],
    )

    result = search_tool.execute(query="anything")

    assert "[Some Course]" in result
    assert search_tool.last_sources == [{"text": "Some Course", "link": "https://example.com/course"}]
    mock_vector_store.get_course_link.assert_called_once_with("Some Course")


def test_execute_dedupes_sources_from_same_lesson(search_tool, mock_vector_store):
    # Real ChromaDB queries commonly return several chunks from the same
    # lesson (different chunk_index, same course_title/lesson_number).
    mock_vector_store.search.return_value = SearchResults(
        documents=["chunk A", "chunk B", "chunk C"],
        metadata=[
            {"course_title": "Some Course", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Some Course", "lesson_number": 0, "chunk_index": 1},
            {"course_title": "Some Course", "lesson_number": 0, "chunk_index": 2},
        ],
        distances=[0.1, 0.15, 0.2],
    )

    result = search_tool.execute(query="anything")

    # All chunks still make it into the formatted context for the LLM...
    assert result.count("[Some Course - Lesson 0]") == 3
    # ...but the UI-facing source list should list that lesson once, not 3x.
    assert search_tool.last_sources == [
        {"text": "Some Course - Lesson 0", "link": "https://example.com/lesson"}
    ]


def test_last_sources_reset_between_calls(search_tool, mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results
    search_tool.execute(query="first call")
    assert len(search_tool.last_sources) == 2

    mock_vector_store.search.return_value = SearchResults(
        documents=["only one result"],
        metadata=[{"course_title": "Other Course", "lesson_number": 1}],
        distances=[0.1],
    )
    search_tool.execute(query="second call")

    assert len(search_tool.last_sources) == 1
    assert search_tool.last_sources[0]["text"] == "Other Course - Lesson 1"
