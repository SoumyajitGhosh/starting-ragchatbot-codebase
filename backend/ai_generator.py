import json
from openai import OpenAI
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with OpenAI's API for generating responses"""

    # Maximum number of sequential tool-calling rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content/detailed educational topics (e.g. "what does lesson 3 say about X")
- **get_course_outline**: Get a course's title, course link, and full lesson list (lesson number + title for each lesson) for questions about course structure, syllabus, or "what lessons are in course X"

Tool Usage:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about a course's outline, structure, syllabus, or lesson list
- When a question names a specific lesson number (e.g. "lesson 2", "the third lesson"), you MUST pass it via the `lesson_number` parameter of search_course_content, not as part of `query` - omitting it skips lesson filtering and searches the whole course
- You may use tools across up to two sequential rounds per question when one call's results are needed to inform the next. Example: call get_course_outline to find a lesson's title, then use that title as the search topic in a second call to search_course_content on a different course.
- Only make a second tool call if it's genuinely needed to answer the question - do not repeat a call with the same effective parameters, and do not chain calls "just in case"
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use the appropriate tool first, then answer
- **Outline/structure questions**: When using get_course_outline, your answer must include the course title, the course link, and the full lesson list with each lesson's number and title, exactly as returned by the tool
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS rounds of tool calling: each round is a
        separate API request, so the model can reason about a prior tool result
        before deciding whether to make another tool call.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        messages = self._build_initial_messages(query, conversation_history)

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._call_model(messages, tools)
            choice = response.choices[0]
            message = choice.message

            # Terminate: no tool call requested (or no tool_manager to run one)
            if (
                choice.finish_reason != "tool_calls"
                or not message.tool_calls
                or not tool_manager
            ):
                return message.content

            messages.append(message)

            error = self._execute_tool_calls(message.tool_calls, messages, tool_manager)
            if error is not None:
                return error

        # Round budget exhausted but the model still wants to act - force a final,
        # tools-free answer using everything accumulated so far.
        final_response = self._call_model(messages, tools=None)
        return final_response.choices[0].message.content

    def _build_initial_messages(
        self, query: str, conversation_history: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build the initial system/user message list for a new query."""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

    def _call_model(self, messages: List[Dict[str, Any]], tools: Optional[List]):
        """Make one chat completion call, offering tools if provided."""
        api_params = {**self.base_params, "messages": messages}
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        return self.client.chat.completions.create(**api_params)

    def _execute_tool_calls(
        self, tool_calls, messages: List[Dict[str, Any]], tool_manager
    ) -> Optional[str]:
        """
        Execute each tool call for the current round, appending a tool-result
        message per call to `messages`.

        Returns:
            None on success, or a user-facing error string on the first failure
            (remaining calls in this round are not attempted).
        """
        for tool_call in tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, **tool_args
                )
            except Exception:
                return (
                    f"I ran into a problem while using the '{tool_call.function.name}' tool, "
                    "so I can't finish that request right now."
                )

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
            )

        return None
