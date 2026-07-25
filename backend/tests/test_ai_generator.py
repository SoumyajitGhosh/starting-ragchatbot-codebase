"""Tests for AIGenerator's OpenAI tool-calling flow.

Most tests mock the OpenAI client for fast, deterministic coverage of the
tool-calling logic. `test_live_tool_call_round_trip_for_content_question`
hits the real OpenAI API + real ChromaDB to catch real request/response-shape
bugs that mocks can't see (e.g. how the raw assistant message object gets
serialized back into the follow-up request).
"""

import pytest


SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_course_content",
        "description": "Search course materials",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def test_no_tools_no_tool_call_attempted(patched_ai_generator, make_chat_completion):
    generator, mock_client = patched_ai_generator
    mock_client.chat.completions.create.return_value = make_chat_completion(
        content="Paris is the capital of France.", finish_reason="stop"
    )

    result = generator.generate_response(query="What is the capital of France?")

    assert result == "Paris is the capital of France."
    assert mock_client.chat.completions.create.call_count == 1
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


def test_tools_provided_sets_tool_choice_auto(patched_ai_generator, make_chat_completion):
    generator, mock_client = patched_ai_generator
    mock_client.chat.completions.create.return_value = make_chat_completion(
        content="answer", finish_reason="stop"
    )

    generator.generate_response(query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=None)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == [SEARCH_TOOL_DEF]
    assert call_kwargs["tool_choice"] == "auto"


def test_finish_reason_stop_skips_tool_execution(patched_ai_generator, make_chat_completion, mock_tool_manager):
    generator, mock_client = patched_ai_generator
    mock_client.chat.completions.create.return_value = make_chat_completion(
        content="answer", finish_reason="stop"
    )

    result = generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    mock_tool_manager.execute_tool.assert_not_called()
    assert result == "answer"
    assert mock_client.chat.completions.create.call_count == 1


def test_tool_call_triggers_execution_and_second_call(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.return_value = "[Course - Lesson 1]\nsome content"

    tool_call = make_tool_call("call_123", "search_course_content", {"query": "what is X", "course_name": "MCP"})
    first_response = make_chat_completion(content=None, tool_calls=[tool_call], finish_reason="tool_calls")
    second_response = make_chat_completion(content="Final answer using tool results.", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="What is X in MCP course?", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="what is X", course_name="MCP"
    )
    assert result == "Final answer using tool results."
    assert mock_client.chat.completions.create.call_count == 2

    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["tools"] == [SEARCH_TOOL_DEF]
    assert second_call_kwargs["tool_choice"] == "auto"

    messages = second_call_kwargs["messages"]
    tool_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_123"
    assert tool_messages[0]["content"] == "[Course - Lesson 1]\nsome content"


def test_multiple_tool_calls_all_executed(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.side_effect = ["result1", "result2"]

    tool_call_1 = make_tool_call("call_1", "search_course_content", {"query": "a"})
    tool_call_2 = make_tool_call("call_2", "get_course_outline", {"course_name": "MCP"})
    first_response = make_chat_completion(tool_calls=[tool_call_1, tool_call_2], finish_reason="tool_calls")
    second_response = make_chat_completion(content="combined answer", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    assert mock_tool_manager.execute_tool.call_count == 2
    mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="a")
    mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")

    second_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_1", "call_2"}
    assert result == "combined answer"


def test_second_round_still_offers_tools(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.return_value = "some result"

    tool_call = make_tool_call("call_1", "search_course_content", {"query": "a"})
    first_response = make_chat_completion(tool_calls=[tool_call], finish_reason="tool_calls")
    second_response = make_chat_completion(content="answer", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["tools"] == [SEARCH_TOOL_DEF]
    assert second_call_kwargs["tool_choice"] == "auto"


def test_two_tool_rounds_then_forced_final_answer_without_tools(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.side_effect = ["outline result", "search result"]

    round1_call = make_tool_call("call_1", "get_course_outline", {"course_name": "X"})
    round2_call = make_tool_call("call_2", "search_course_content", {"query": "lesson 4 topic"})
    round1_response = make_chat_completion(tool_calls=[round1_call], finish_reason="tool_calls")
    round2_response = make_chat_completion(tool_calls=[round2_call], finish_reason="tool_calls")
    round3_response = make_chat_completion(content="final synthesized answer", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [round1_response, round2_response, round3_response]

    result = generator.generate_response(
        query="find a course on the same topic as lesson 4 of X",
        tools=[SEARCH_TOOL_DEF],
        tool_manager=mock_tool_manager,
    )

    assert mock_client.chat.completions.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2
    mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="X")
    mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 4 topic")

    third_call_kwargs = mock_client.chat.completions.create.call_args_list[2].kwargs
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs

    assert result == "final synthesized answer"


def test_round_cap_forces_final_answer_even_if_model_still_wants_a_tool_call(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.side_effect = ["outline result", "search result"]

    round1_call = make_tool_call("call_1", "get_course_outline", {"course_name": "X"})
    round2_call = make_tool_call("call_2", "search_course_content", {"query": "lesson 4 topic"})
    round1_response = make_chat_completion(tool_calls=[round1_call], finish_reason="tool_calls")
    round2_response = make_chat_completion(tool_calls=[round2_call], finish_reason="tool_calls")
    # Pathological: round 3 still looks tool-call-shaped, but no tools were offered on
    # that call, so the loop must not try to execute a third round regardless.
    another_call = make_tool_call("call_3", "search_course_content", {"query": "should not run"})
    round3_response = make_chat_completion(
        content="best-effort answer", tool_calls=[another_call], finish_reason="tool_calls"
    )
    mock_client.chat.completions.create.side_effect = [round1_response, round2_response, round3_response]

    result = generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    assert mock_client.chat.completions.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2
    assert result == "best-effort answer"


def test_tool_execution_exception_returns_graceful_message(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.side_effect = RuntimeError("vector store unavailable")

    tool_call = make_tool_call("call_1", "search_course_content", {"query": "a"})
    first_response = make_chat_completion(tool_calls=[tool_call], finish_reason="tool_calls")
    mock_client.chat.completions.create.return_value = first_response

    result = generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    assert isinstance(result, str)
    assert result.strip() != ""
    assert mock_client.chat.completions.create.call_count == 1
    assert mock_tool_manager.execute_tool.call_count == 1


def test_tool_error_string_result_is_not_treated_as_failure(
    patched_ai_generator, make_chat_completion, make_tool_call, mock_tool_manager
):
    generator, mock_client = patched_ai_generator
    mock_tool_manager.execute_tool.return_value = "No relevant content found."

    tool_call = make_tool_call("call_1", "search_course_content", {"query": "a"})
    first_response = make_chat_completion(tool_calls=[tool_call], finish_reason="tool_calls")
    second_response = make_chat_completion(content="answer despite no results", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="anything", tools=[SEARCH_TOOL_DEF], tool_manager=mock_tool_manager
    )

    assert mock_client.chat.completions.create.call_count == 2
    second_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "No relevant content found."
    assert result == "answer despite no results"


@pytest.mark.integration
def test_lesson_number_extracted_into_tool_call_args(rag_system, monkeypatch):
    """A question naming a specific lesson must reach the tool as lesson_number=N,
    not just as text inside the free-text `query` (which skips the lesson filter
    entirely and searches the whole course)."""
    captured_calls = []
    original_execute_tool = rag_system.tool_manager.execute_tool

    def spy_execute_tool(tool_name, **kwargs):
        captured_calls.append((tool_name, kwargs))
        return original_execute_tool(tool_name, **kwargs)

    monkeypatch.setattr(rag_system.tool_manager, "execute_tool", spy_execute_tool)

    rag_system.ai_generator.generate_response(
        query="What is covered in lesson 2 of the MCP course?",
        tools=rag_system.tool_manager.get_tool_definitions(),
        tool_manager=rag_system.tool_manager,
    )

    assert len(captured_calls) == 1
    tool_name, kwargs = captured_calls[0]
    assert tool_name == "search_course_content"
    assert kwargs.get("lesson_number") == 2, (
        f"Expected lesson_number=2 in the tool call, got kwargs={kwargs!r}. "
        "The model likely put the lesson number in `query` instead."
    )


@pytest.mark.integration
def test_live_tool_call_round_trip_for_content_question(rag_system):
    """Exercises the real OpenAI tool-calling round trip end to end.

    This is the one place a raw-message-serialization bug from the OpenAI
    migration would actually surface, since mocked tests never send the
    assistant message object through the real SDK/HTTP layer.
    """
    ai_generator = rag_system.ai_generator
    tool_manager = rag_system.tool_manager

    result = ai_generator.generate_response(
        query="What does lesson 0 of the Building Towards Computer Use with Anthropic course cover?",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    assert isinstance(result, str)
    assert result.strip() != ""

    sources = tool_manager.get_last_sources()
    tool_manager.reset_sources()
    assert len(sources) > 0
