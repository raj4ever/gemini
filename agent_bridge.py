"""Bridge OpenAI tool-calling (Hermes Agent) to text-based Gemini cookie API."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

AGENT_MAX_TOOLS = int(os.environ.get("GEMINI_AGENT_MAX_TOOLS", "30"))
AGENT_MAX_MESSAGES = int(os.environ.get("GEMINI_AGENT_MAX_MESSAGES", "24"))
AGENT_MAX_MSG_CHARS = int(os.environ.get("GEMINI_AGENT_MAX_MSG_CHARS", "12000"))
AGENT_MAX_TOOL_DESC = int(os.environ.get("GEMINI_AGENT_MAX_TOOL_DESC", "280"))

TOOL_CALL_TAG = "hermes_tool_call"
TOOL_CALL_BLOCK_PATTERN = re.compile(
    rf"<{TOOL_CALL_TAG}>(.*?)</{TOOL_CALL_TAG}>",
    re.DOTALL,
)

AGENT_SYSTEM_PREFIX = """# Agent mode (mandatory tool protocol)

You control a real computer via tools. You MUST execute actions with tools — never only describe what you would do.

When you need to run commands, read/write files, search, or use any capability below:
1. Output one or more tool calls in this EXACT format (valid JSON inside tags):
<hermes_tool_call>
{"name": "TOOL_NAME", "arguments": {"key": "value"}}
</hermes_tool_call>
2. Use ONLY tool names from the list below.
3. `arguments` must be a JSON object matching the tool's parameters.
4. You may add a short sentence before tool calls; the tool block is required for any action.
5. After you receive [Tool result] messages, continue until the task is done.
6. When the task is fully complete with no more tools needed, reply normally WITHOUT any <hermes_tool_call> tags.

"""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]"


def _compact_tools_for_prompt(
    tools: list[dict[str, Any]], max_tools: int = AGENT_MAX_TOOLS
) -> str:
    """Serialize tool definitions for the prompt (cap count to avoid huge prompts)."""
    compact: list[dict[str, Any]] = []
    for t in tools[:max_tools]:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn:
            continue
        entry: dict[str, Any] = {
            "name": fn.get("name", ""),
            "description": (fn.get("description") or "")[:AGENT_MAX_TOOL_DESC],
        }
        params = fn.get("parameters")
        if params:
            entry["parameters"] = params
        compact.append(entry)
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _format_tool_calls_for_history(tool_calls: list[Any]) -> str:
    lines: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args = getattr(fn, "arguments", "{}") if fn else "{}"
        if isinstance(args, str):
            try:
                args_obj = json.loads(args)
            except json.JSONDecodeError:
                args_obj = {"raw": args}
        else:
            args_obj = args if isinstance(args, dict) else {"value": args}
        payload = json.dumps({"name": name, "arguments": args_obj}, ensure_ascii=False)
        lines.append(f"<{TOOL_CALL_TAG}>\n{payload}\n</{TOOL_CALL_TAG}>")
    return "\n".join(lines)


def extract_message_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif "text" in block:
                parts.append(str(block["text"]))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def _trim_messages_for_agent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep recent turns only — long Hermes sessions hang VPS Gemini."""
    if len(messages) <= AGENT_MAX_MESSAGES:
        trimmed = messages
    else:
        head = [m for m in messages if m.get("role") == "system"][:2]
        tail = messages[-AGENT_MAX_MESSAGES:]
        trimmed = head + [m for m in tail if m not in head]
    out: list[dict[str, Any]] = []
    for msg in trimmed:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str) and len(content) > AGENT_MAX_MSG_CHARS:
            m["content"] = _truncate(content, AGENT_MAX_MSG_CHARS)
        out.append(m)
    return out


def messages_to_agent_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
) -> str:
    """Build a single prompt including tools, history, and tool results."""
    messages = _trim_messages_for_agent(messages)
    parts: list[str] = []

    if tools:
        parts.append(AGENT_SYSTEM_PREFIX)
        parts.append("# Available tools (JSON)\n")
        parts.append(_compact_tools_for_prompt(tools))
        if tool_choice == "required":
            parts.append(
                "\n# This turn: you MUST call at least one tool before finishing.\n"
            )

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            text = extract_message_text(msg.get("content"))
            if text.strip():
                parts.append(f"[System]\n{text}")
        elif role == "user":
            text = extract_message_text(msg.get("content"))
            if text.strip():
                parts.append(f"[User]\n{text}")
        elif role == "assistant":
            text = extract_message_text(msg.get("content"))
            tcs = msg.get("tool_calls")
            block = "[Assistant]"
            if text.strip():
                block += f"\n{text}"
            if tcs:
                block += "\n" + _format_tool_calls_for_history(tcs)
            if block != "[Assistant]":
                parts.append(block)
        elif role == "tool":
            name = msg.get("name") or "tool"
            tid = msg.get("tool_call_id") or ""
            text = extract_message_text(msg.get("content"))
            parts.append(f"[Tool result — {name} (id={tid})]\n{text}")
        else:
            text = extract_message_text(msg.get("content"))
            if text.strip():
                parts.append(f"[{role}]\n{text}")

    parts.append("[Assistant]")
    return "\n\n".join(parts)


def _append_tool_call(tool_calls: list[dict[str, Any]], data: dict[str, Any]) -> None:
    name = data.get("name")
    if not name:
        return
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"value": args}
    tool_calls.append(
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": str(name),
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
    )


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract tool calls from model output; return cleaned text + OpenAI tool_calls list."""
    tool_calls: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last_end = 0

    for match in TOOL_CALL_BLOCK_PATTERN.finditer(text):
        cleaned_parts.append(text[last_end : match.start()])
        raw = match.group(1).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                _append_tool_call(tool_calls, data)
            except json.JSONDecodeError:
                pass
        last_end = match.end()

    cleaned_parts.append(text[last_end:])
    cleaned = "".join(cleaned_parts).strip()

    if not tool_calls:
        for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            if "name" in data:
                _append_tool_call(tool_calls, data)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, tool_calls


def build_completion_message(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    content = text if text else None
    if tool_calls:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
    return {"role": "assistant", "content": content or ""}


def build_choice(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return {
        "index": 0,
        "message": build_completion_message(text, tool_calls),
        "finish_reason": finish_reason,
    }


def _sse_chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any] | None,
    finish_reason: str | None = None,
) -> str:
    import time

    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def build_stream_tool_chunks(
    completion_id: str,
    model: str,
    tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Emit OpenAI-style streaming chunks for tool_calls (after buffering full response)."""
    chunks: list[str] = []
    for i, tc in enumerate(tool_calls):
        fn = tc["function"]
        chunks.append(
            _sse_chunk(
                completion_id,
                model,
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": fn["name"], "arguments": ""},
                        }
                    ]
                },
            )
        )
        chunks.append(
            _sse_chunk(
                completion_id,
                model,
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "function": {"arguments": fn["arguments"]},
                        }
                    ]
                },
            )
        )
    chunks.append(_sse_chunk(completion_id, model, {}, finish_reason="tool_calls"))
    chunks.append("data: [DONE]\n\n")
    return chunks
