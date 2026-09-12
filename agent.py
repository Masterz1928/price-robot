import os
import json
from dotenv import load_dotenv
from groq import Groq
import groq

from tools import search_shopee, search_lazada

load_dotenv()

client = Groq(api_key=os.environ["GROQ"])

MODEL = "openai/gpt-oss-20b"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are a price comparison agent. Your job is to find the
best deal for a product the user asks about, by searching Shopee and Lazada.

You ONLY have access to search_shopee and search_lazada. Do NOT attempt to
call any other tool - you cannot open URLs or visit pages directly.

STRICT RULE: Search each site AT MOST ONCE, unless that search returned ZERO
results (in which case you may retry that one site ONCE with a broader query).
As soon as you have results from EACH site (or have tried twice and gotten
nothing), STOP searching and give your final answer immediately.

IMPORTANT: "Best deal" does NOT simply mean cheapest price. The user will give
you requirements - some are FIXED/objective (e.g. "bluetooth") and some are
SUBJECTIVE/fuzzy (e.g. "must be small"). For every candidate product, evaluate
it against EACH requirement individually.

When giving your final answer, you MUST respond with ONLY valid JSON matching
this exact schema, and nothing else - no markdown, no extra text, no code fences:

{
  "products": [
    {
      "name": "string - product name",
      "site": "string - Shopee or Lazada",
      "link": "string - the product URL",
      "price": "string - price with currency, or 'Unknown' if not found",
      "requirements": [
        {
          "label": "string - the requirement being evaluated",
          "type": "string - either 'fixed' or 'fuzzy'",
          "score": "number 0-5 - how well this listing satisfies it (0=fails, 5=perfect match)",
          "note": "string - short reason based on the actual listing content"
        }
      ],
      "description": "string - 1 sentence summary of this product"
    }
  ],
  "recommendation": "string - your final recommendation and reasoning, concise",
  "search_suggestions": "string or null - if results were poor/zero, suggest 2-3 alternative search terms here, otherwise null"
}

For FIXED requirements, score should effectively be binary: 5 if clearly met,
0 if clearly not met, 2 if unclear/not mentioned in the listing.
For FUZZY requirements, use the full 0-5 range based on how well the listing
description supports it - don't just guess, base it on real listing content.
If a listing doesn't provide enough info to judge a requirement, score it low
(1-2) and say so honestly in the note rather than assuming it passes.
"""
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_shopee",
            "description": "Search Shopee Malaysia for a product and get prices/listings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_lazada",
            "description": "Search Lazada Malaysia for a product and get prices/listings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product to search for"}
                },
                "required": ["query"],
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "search_shopee": search_shopee,
    "search_lazada": search_lazada,
}

def format_raw_fallback(messages):
    """Extract raw tool results from message history as a last-resort answer."""
    raw_results = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                data = json.loads(msg["content"])
                raw_results.extend(data if isinstance(data, list) else [])
            except (json.JSONDecodeError, TypeError):
                continue

    if not raw_results:
        return "No results found. Try a different or more specific search term."

    lines = ["Here's what I found (raw results, comparison unavailable):\n"]
    for r in raw_results[:5]:
        lines.append(f"- {r.get('title', 'Unknown')} — {r.get('url', '')}")
    return "\n".join(lines)


def get_available_tools(sites_with_results):
    """Only offer tools for sites we haven't gotten results from yet."""
    return [
        tool for tool in TOOL_DEFINITIONS
        if tool["function"]["name"] not in sites_with_results
    ]

def run_agent(user_request: str, on_step=None):
    def emit(step_type, content):
        if on_step:
            on_step({"type": step_type, "content": content})
        print(f"[{step_type.upper()}] {content}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    sites_with_results = set()

    for iteration in range(MAX_ITERATIONS):
        emit("info", f"--- Iteration {iteration} ---")

        force_final = (
            iteration == MAX_ITERATIONS - 1
            or len(sites_with_results) >= 2
        )

        try:
            if force_final:
                messages.append({
                    "role": "user",
                    "content": "You must now give your final answer as the JSON schema described. Do not attempt to call any tools - none are available."
                })
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            else:
                available_tools = get_available_tools(sites_with_results)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                )
        except groq.BadRequestError as e:
            if force_final:
                emit("warning", f"Couldn't generate a clean summary: {e}")
                fallback_text = format_raw_fallback(messages)
                fallback_obj = {"products": [], "recommendation": fallback_text, "search_suggestions": None}
                emit("final_answer", fallback_obj)
                return fallback_obj
            emit("warning", f"Model attempted an invalid action: {e}")
            messages.append({
                "role": "user",
                "content": "That action isn't available. Only use search_shopee or search_lazada, or give your final answer as plain text.",
            })
            continue

        message = response.choices[0].message

        # Case 1: model wants to call a tool
        if message.tool_calls:
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                emit("thinking", f"Calling {tool_name}({tool_args.get('query', '')})")

                tool_fn = AVAILABLE_TOOLS.get(tool_name)
                result = tool_fn(**tool_args) if tool_fn else {"error": "unknown tool"}

                emit("observation", f"Got {len(result)} result(s) from {tool_name}")

                if isinstance(result, list) and len(result) > 0:
                    sites_with_results.add(tool_name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            continue  # loop again

        # Case 2: no tool calls - this is the final answer
        if force_final:
            try:
                parsed = json.loads(message.content)
                emit("final_answer", parsed)
                return parsed
            except json.JSONDecodeError:
                emit("warning", "Model didn't return valid JSON, falling back to raw results.")
                fallback_text = format_raw_fallback(messages)
                fallback_obj = {"products": [], "recommendation": fallback_text, "search_suggestions": None}
                emit("final_answer", fallback_obj)
                return fallback_obj
        else:
            emit("final_answer", message.content)
            return message.content

    emit("final_answer", {"products": [], "recommendation": "Reached max reasoning steps without a confident answer.", "search_suggestions": None})
    return None

if __name__ == "__main__":
    query = input("What product should I find the best price for? ")
    run_agent(query)