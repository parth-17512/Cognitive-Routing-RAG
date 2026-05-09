"""
phase2_content_engine.py
────────────────────────
Phase 2 — The Autonomous Content Engine (LangGraph)

Graph structure
───────────────

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        LangGraph State Machine                      │
  │                                                                     │
  │  [START] → [decide_search] → [web_search] → [draft_post] → [END]  │
  └─────────────────────────────────────────────────────────────────────┘

Node 1 — decide_search
    The LLM reads the bot's persona and decides what topic it wants to post
    about today, then formats a concise search query string.

Node 2 — web_search
    Calls the mock_searxng_search tool with the query from Node 1 and
    returns headline-style context.

Node 3 — draft_post
    The LLM receives the bot's persona (system prompt) + search results
    (context) and produces a strictly structured JSON post:
        {"bot_id": "...", "topic": "...", "post_content": "..."}
    Enforced via Pydantic schema + JSON mode.
"""

import json
import re
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from config import BOT_PERSONAS
from llm_provider import get_llm


# ─────────────────────────────────────────────────────────────────────────────
# Mock Search Tool
# ─────────────────────────────────────────────────────────────────────────────

# Keyword → headline mapping (simulates a real SearXNG / Brave search)
MOCK_HEADLINES: dict[str, str] = {
    "ai":      "OpenAI releases GPT-5 with autonomous reasoning; rivals scramble to respond.",
    "openai":  "OpenAI releases GPT-5 with autonomous reasoning; rivals scramble to respond.",
    "llm":     "Meta open-sources Llama-4 as the AI arms race accelerates globally.",
    "crypto":  "Bitcoin hits new all-time high amid regulatory ETF approvals and institutional inflows.",
    "bitcoin": "Bitcoin hits new all-time high amid regulatory ETF approvals and institutional inflows.",
    "space":   "SpaceX Starship completes first fully successful orbital flight; Moon mission on track.",
    "elon":    "Elon Musk announces xAI supercomputer cluster; claims AGI within 18 months.",
    "tech":    "Silicon Valley layoffs reach 200,000 in 2025 as AI replaces entry-level roles.",
    "stock":   "S&P 500 hits record high as Fed signals rate cuts; tech sector leads rally.",
    "market":  "S&P 500 hits record high as Fed signals rate cuts; tech sector leads rally.",
    "rate":    "Federal Reserve holds interest rates steady; analysts forecast two cuts by year-end.",
    "privacy": "EU fines Meta €1.2 billion for GDPR violations; activists demand stronger enforcement.",
    "climate": "Record-breaking heatwaves across Europe prompt emergency climate legislation debate.",
    "social":  "New study links TikTok algorithm to increased teen anxiety and reduced attention spans.",
    "default": "Breaking: Global tech leaders convene at Davos to discuss AI governance frameworks.",
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search.

    Parameters
    ----------
    query : str
        A short search query string.

    Returns
    -------
    str
        A hardcoded recent headline relevant to the query keywords.
    """
    q_lower = query.lower()
    for keyword, headline in MOCK_HEADLINES.items():
        if keyword in q_lower:
            return headline
    return MOCK_HEADLINES["default"]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    bot_id: str               # e.g. "bot_a"
    persona: str              # full persona string
    search_query: str         # populated by Node 1
    search_results: str       # populated by Node 2
    final_post: dict          # populated by Node 3


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schema — enforces structured JSON output from Node 3
# ─────────────────────────────────────────────────────────────────────────────

class BotPost(BaseModel):
    """Structured output schema for a bot-generated post."""
    bot_id: str = Field(description="The unique identifier of the bot (e.g. 'bot_a')")
    topic: str = Field(description="The topic or theme of the post in 3-6 words")
    post_content: str = Field(
        description="The tweet-style post, maximum 280 characters, highly opinionated and in-persona"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def node_decide_search(state: GraphState) -> GraphState:
    """
    Node 1 — Decide Search

    The LLM reads the bot's persona and decides what topic to post about
    today, then outputs a concise search query string.
    """
    print(f"\n[Node 1 — decide_search] Bot: {state['bot_id']}")

    llm = get_llm(temperature=0.7)

    messages = [
        SystemMessage(
            content=(
                "You are the following social media bot persona:\n\n"
                f"{state['persona']}\n\n"
                "Your job is to decide what topic you want to post about TODAY based on your persona. "
                "Output ONLY a short search query (3–7 words) that you would type into a search engine "
                "to find relevant news. Do NOT explain. Output the query string only."
            )
        ),
        HumanMessage(content="What do you want to post about today? Give me your search query."),
    ]

    response = llm.invoke(messages)
    query = response.content.strip().strip('"').strip("'")
    print(f"  → Search query decided: \"{query}\"")

    return {**state, "search_query": query}


def node_web_search(state: GraphState) -> GraphState:
    """
    Node 2 — Web Search

    Calls the mock_searxng_search tool with the query from Node 1.
    """
    print(f"\n[Node 2 — web_search] Query: \"{state['search_query']}\"")

    # Call the tool directly (LangGraph tool invocation)
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  → Search result: \"{result}\"")

    return {**state, "search_results": result}


def node_draft_post(state: GraphState) -> GraphState:
    """
    Node 3 — Draft Post

    The LLM uses its persona (system prompt) + search results (context) to
    generate a highly opinionated 280-character post.
    Structured output is enforced via Pydantic's .with_structured_output().
    """
    print(f"\n[Node 3 — draft_post] Generating post for {state['bot_id']} …")

    # Use structured output mode — guarantees BotPost JSON schema
    llm = get_llm(temperature=0.85)
    structured_llm = llm.with_structured_output(BotPost)

    messages = [
        SystemMessage(
            content=(
                "You are the following social media bot:\n\n"
                f"Bot ID: {state['bot_id']}\n"
                f"Persona: {state['persona']}\n\n"
                "RULES:\n"
                "1. Write a post that reflects your persona STRONGLY — be opinionated.\n"
                "2. Incorporate the provided news headline as context or evidence.\n"
                "3. Keep post_content under 280 characters.\n"
                "4. Do NOT use hashtags.\n"
                "5. Sound authentic to your persona, not like a generic AI."
            )
        ),
        HumanMessage(
            content=(
                f"Today's news context:\n\"{state['search_results']}\"\n\n"
                "Now write your post using the schema provided."
            )
        ),
    ]

    post: BotPost = structured_llm.invoke(messages)

    # Ensure bot_id matches (LLM sometimes hallucinates a different ID)
    post_dict = post.model_dump()
    post_dict["bot_id"] = state["bot_id"]

    print(f"  → Post drafted:")
    print(f"     bot_id      : {post_dict['bot_id']}")
    print(f"     topic       : {post_dict['topic']}")
    print(f"     post_content: {post_dict['post_content']}")

    return {**state, "final_post": post_dict}


# ─────────────────────────────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_content_graph() -> StateGraph:
    """Assembles and compiles the LangGraph state machine."""
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search", node_web_search)
    graph.add_node("draft_post", node_draft_post)

    # Wire edges: START → decide_search → web_search → draft_post → END
    graph.add_edge(START, "decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_bot_post(bot_id: str) -> dict:
    """
    Run the full LangGraph pipeline for a given bot and return the structured
    JSON post.

    Parameters
    ----------
    bot_id : str
        One of "bot_a", "bot_b", "bot_c".

    Returns
    -------
    dict
        {"bot_id": "...", "topic": "...", "post_content": "..."}
    """
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id '{bot_id}'. Valid: {list(BOT_PERSONAS.keys())}")

    persona = BOT_PERSONAS[bot_id]["persona"]

    initial_state: GraphState = {
        "bot_id": bot_id,
        "persona": persona,
        "search_query": "",
        "search_results": "",
        "final_post": {},
    }

    app = build_content_graph()
    final_state = app.invoke(initial_state)
    return final_state["final_post"]


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for bot_id in BOT_PERSONAS:
        print(f"\n{'═'*60}")
        print(f"Running content engine for: {bot_id}")
        print(f"{'═'*60}")
        result = generate_bot_post(bot_id)
        print(f"\n✅ Final JSON output:")
        print(json.dumps(result, indent=2))
