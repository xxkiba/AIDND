# agent_workflow.py
"""
ReACT-style workflow/orchestrator.

- Holds the system prompt (English)
- Parses <CALL>{...}</CALL> blocks from the model's output
- Dispatches to tool functions defined in aidnd_tools.py
- Feeds tool "Observation" back to the model
- Returns the final, user-facing answer

You should implement `call_llm()` with your model provider.
"""
from openai import OpenAI
import json
import re
import os
from typing import Dict, Any, List


from aidnd_tools import (
    look_monster_table,
    search_table,
    fetch_and_cache,
)

# ================== Plug in your LLM here ==================
def call_llm(messages):
    """
    Minimal OpenAI GPT-4o call compatible with the ReACT workflow.

    Requirements:
      - Install the official client:  pip install openai>=1.0.0
      - Set your key:  export OPENAI_API_KEY="sk-..."
        (or set it in your environment variables on Windows)

    Parameters
    ----------
    messages : list[dict]
        Conversation so far, in the format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns
    -------
    str : The assistant's text reply.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ================== System prompt (English) ==================
SYSTEM_PROMPT = (
    "[You are AIDND Assistant]\n"
    "You MUST follow this ReACT tool-calling protocol. When you need data from the local catalog or Open5e, you MUST call tools.\n"
    "Do NOT narrate that you are 'going to look things up'. Instead, output exactly one tool call block:\n"
    "  <CALL>{\"fn\":\"function_name\",\"args\":{...}}</CALL>\n"
    "After the system runs the tool, it will append a system message starting with 'Observation: { ... }'.\n"
    "You may think again, optionally call more tools, and ONLY AFTER you have fetched details with fetch_and_cache, produce a final user-facing answer.\n"
    "Never include <CALL> in your final answer.\n"
    "\n"
    "Available functions:\n"
    "- look_monster_table(query:str, limit:int=20)\n"
    "  Search the local monster name->slug index; returns candidate names with slug lists.\n"
    "- search_table(type:str, name_or_slug:str, prefer_doc:str|None)\n"
    "  Resolve to a unique entry (e.g., pick a specific slug) and return its api_url.\n"
    "- fetch_and_cache(type:str, slug:str)\n"
    "  Fetch detailed JSON from Open5e using the resolved slug (with local caching).\n"
    "\n"
    "Typical flow:\n"
    "  user question -> look_monster_table (find candidates) -> search_table (disambiguate) ->\n"
    "  fetch_and_cache (get details) -> final explanation.\n"
    "\n"
    "Few-shot example (format illustration):\n"
    "User: Tell me about Zombie.\n"
    "Assistant: <CALL>{\"fn\":\"look_monster_table\",\"args\":{\"query\":\"Zombie\",\"limit\":10}}</CALL>\n"
    "System: Observation: {\"fn\":\"look_monster_table\", \"result\": {\"matches\": [{\"name\":\"Zombie\",\"slugs\":[\"zombie\",\"zombie-a5e\"]}]}}\n"
    "Assistant: <CALL>{\"fn\":\"search_table\",\"args\":{\"type\":\"monsters\",\"name_or_slug\":\"Zombie\",\"prefer_doc\":\"srd-2014\"}}</CALL>\n"
    "System: Observation: {\"fn\":\"search_table\", \"result\": {\"chosen_name\":\"Zombie\",\"chosen_slug\":\"zombie\",\"api_url\":\"...\"}}\n"
    "Assistant: <CALL>{\"fn\":\"fetch_and_cache\",\"args\":{\"type\":\"monsters\",\"slug\":\"zombie\"}}</CALL>\n"
    "System: Observation: {\"fn\":\"fetch_and_cache\", \"result\": {\"slug\":\"zombie\",\"data\": {\"name\":\"Zombie\",\"size\":\"Medium\", \"hit_points\":22, ...}}}\n"
    "Assistant: (final answer to user summarizing the fetched JSON; no more CALL blocks.)\n"
)


# ================== CALL parsing & dispatch ==================
CALL_RE = re.compile(r"<CALL>([\s\S]+?)</CALL>", re.IGNORECASE)

def _maybe_execute_tool(text: str) -> Dict[str, Any] | None:
    """
    Detect a single <CALL>{...}</CALL> in the assistant text.
    If found, parse JSON and dispatch to the corresponding tool.
    Returns a dict with {"fn","args","result"} or {"error":...}, or None if no tool call exists.
    """
    m = CALL_RE.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
        fn  = payload["fn"]
        args = payload.get("args", {})
    except Exception as e:
        return {"error": f"CALL parse error: {e}"}

    try:
        if fn == "look_monster_table":
            result = look_monster_table(
                query=args.get("query", ""),
                limit=int(args.get("limit", 20)),
            )
        elif fn == "search_table":
            result = search_table(
                res_type=args.get("type", "monsters"),
                name_or_slug=args.get("name_or_slug", ""),
                prefer_doc=args.get("prefer_doc"),
            )
        elif fn == "fetch_and_cache":
            result = fetch_and_cache(
                res_type=args.get("type", "monsters"),
                slug=args.get("slug", ""),
            )
        else:
            return {"error": f"unknown tool: {fn}"}
        return {"fn": fn, "args": args, "result": result}
    except Exception as e:
        return {"fn": fn, "args": args, "error": str(e)}


# ================== Public entry: one full turn ==================

import logging
from datetime import datetime
import os

# Setup logger once
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def answer_query(user_query: str, max_tool_steps: int = 4) -> str:
    """
    ReACT main loop with logging to file instead of console.
    Each round (assistant, tool call, observation) is logged to a timestamped .log file.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    logging.info("=" * 80)
    logging.info(f"[USER QUERY] {user_query}")
    logging.info("=" * 80)

    for step in range(max_tool_steps):
        logging.info(f"\n[STEP {step + 1}] Sending messages to model:")
        for m in msgs[-3:]:
            truncated = m["content"][:1000] + ("..." if len(m["content"]) > 1000 else "")
            logging.info(f"  {m['role'].upper()}: {truncated}")

        assistant_text = call_llm(msgs)
        logging.info(f"[MODEL OUTPUT STEP {step + 1}] ----------------------------")
        logging.info(assistant_text)
        logging.info("------------------------------------------------------------")

        msgs.append({"role": "assistant", "content": assistant_text})

        call = _maybe_execute_tool(assistant_text)
        if not call:
            logging.info("[NO TOOL CALL DETECTED] Returning final answer.\n")
            logging.info("=" * 80)
            logging.info(f"[FINAL ANSWER]\n{assistant_text}")
            logging.info("=" * 80)
            return assistant_text

        # Tool call found: execute and log the observation
        logging.info(f"[TOOL CALL DETECTED] {call.get('fn')} with args = {call.get('args', {})}")
        observation_json = json.dumps(call, ensure_ascii=False, indent=2)
        logging.info(f"[OBSERVATION]\n{observation_json}")
        msgs.append({"role": "system", "content": f"Observation: {observation_json}"})

    logging.warning("[TOOL CALL LIMIT REACHED] The model did not finish within the allowed steps.")
    return "Tool call limit reached. Please provide a final answer based on the observations so far."
