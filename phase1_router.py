"""
phase1_router.py
────────────────
Phase 1 — Vector-Based Persona Matching (The Router)

How it works
────────────
1. On startup, each bot's persona text is embedded and stored in a ChromaDB
   in-memory collection.
2. When a post arrives, it is embedded with the same model.
3. Cosine similarity between the post vector and every persona vector is
   computed (ChromaDB does this internally and returns `distance` which is
   1 − cosine_similarity for normalised vectors).
4. Only bots whose similarity exceeds `threshold` are returned.

ChromaDB note
─────────────
ChromaDB's default metric is "cosine" distance (not similarity).
    cosine_similarity = 1 - cosine_distance
So we convert:  similarity = 1 - result["distances"][0][i]
"""

from typing import Optional

import chromadb
from chromadb.config import Settings

from config import BOT_PERSONAS, SIMILARITY_THRESHOLD
from embeddings import get_embeddings

# ─────────────────────────────────────────────────────────────────────────────
# Singleton: build the in-memory vector store once and reuse it.
# ─────────────────────────────────────────────────────────────────────────────
_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None
_embed_model = None


def _build_vector_store() -> chromadb.Collection:
    """
    Embed all bot personas and insert them into an in-memory ChromaDB
    collection.  Called once on first use (lazy initialisation).
    """
    global _chroma_client, _collection, _embed_model

    print("\n[Phase 1] Building in-memory persona vector store …")

    # 1. Instantiate the embedding model
    _embed_model = get_embeddings()

    # 2. Create an in-memory ChromaDB client (no files written to disk)
    _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

    # 3. Create a collection with cosine distance metric
    _collection = _chroma_client.create_collection(
        name="bot_personas",
        metadata={"hnsw:space": "cosine"},  # cosine distance
    )

    # 4. Embed each persona and upsert into the collection
    ids, embeddings, documents, metadatas = [], [], [], []
    for bot_id, bot in BOT_PERSONAS.items():
        persona_text = bot["persona"]
        vector = _embed_model.embed_query(persona_text)

        ids.append(bot_id)
        embeddings.append(vector)
        documents.append(persona_text)
        metadatas.append({"name": bot["name"]})

        print(f"  ✔ Embedded persona for {bot_id} ({bot['name']})")

    _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"[Phase 1] Vector store ready — {len(ids)} personas indexed.\n")
    return _collection


def _get_collection() -> chromadb.Collection:
    """Returns the (lazily initialised) ChromaDB collection."""
    if _collection is None:
        _build_vector_store()
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def route_post_to_bots(
    post_content: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Find which bots "care" about the given post using cosine similarity.

    Parameters
    ----------
    post_content : str
        The raw text of the incoming post.
    threshold : float
        Minimum cosine similarity (0–1) for a bot to be selected.
        Default is loaded from the SIMILARITY_THRESHOLD env variable.

    Returns
    -------
    List of matched bot dicts, each with keys:
        {bot_id, name, persona, similarity}
    Sorted by similarity (highest first).
    """
    collection = _get_collection()

    # Embed the incoming post
    print(f"[Phase 1] Routing post: \"{post_content}\"")
    post_vector = _embed_model.embed_query(post_content)

    # Query ALL bots (n_results = total personas) to rank them
    n_bots = len(BOT_PERSONAS)
    results = collection.query(
        query_embeddings=[post_vector],
        n_results=n_bots,
        include=["distances", "metadatas", "documents"],
    )

    matched_bots: list[dict] = []

    print(f"\n{'─'*55}")
    print(f"{'Bot':<20} {'Similarity':>12} {'Match?':>8}")
    print(f"{'─'*55}")

    for i, bot_id in enumerate(results["ids"][0]):
        # ChromaDB returns cosine *distance*; convert to similarity
        cosine_distance = results["distances"][0][i]
        similarity = 1.0 - cosine_distance

        bot_name = results["metadatas"][0][i]["name"]
        matched = similarity >= threshold

        print(
            f"{bot_id} ({bot_name:<14}) {similarity:>12.4f} {'✅ MATCHED' if matched else '❌ skipped':>8}"
        )

        if matched:
            matched_bots.append(
                {
                    "bot_id": bot_id,
                    "name": bot_name,
                    "persona": BOT_PERSONAS[bot_id]["persona"],
                    "similarity": round(similarity, 4),
                }
            )

    print(f"{'─'*55}")
    print(
        f"[Phase 1] {len(matched_bots)}/{n_bots} bot(s) matched "
        f"(threshold = {threshold}).\n"
    )

    # Return sorted by similarity (highest first)
    return sorted(matched_bots, key=lambda b: b["similarity"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo — run this file directly to test routing
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "New study shows social media algorithms are increasing teen depression rates.",
        "The Fed just raised interest rates by 50 basis points; market reacts sharply.",
    ]

    for post in test_posts:
        print(f"\n{'═'*60}")
        print(f"POST: {post}")
        print(f"{'═'*60}")
        matched = route_post_to_bots(post)
        if matched:
            print("Routed to:")
            for b in matched:
                print(f"  → {b['bot_id']} ({b['name']}) — similarity: {b['similarity']}")
        else:
            print("  No bots matched this post (below threshold).")
