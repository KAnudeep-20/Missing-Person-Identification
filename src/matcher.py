"""Cosine-similarity matching between a query embedding and the reference database.

Because every embedding is L2-normalized, cosine similarity reduces to a dot
product (spec section 13).

Public API (spec section 26):
    match_embedding(query_embedding, reference_database, threshold)
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(query_embedding, reference_embedding) -> float:
    """Dot product of two already-normalized embeddings."""
    return float(np.dot(query_embedding, reference_embedding))


def match_embedding(query_embedding, reference_database: dict, threshold: float) -> dict:
    """Find the best candidate for a query embedding.

    Compares the query against every reference embedding (spec section 12,
    option 1: highest single-image similarity wins).

    Returns:
        {
            "person_id": best candidate id or None,
            "similarity": best cosine similarity,
            "matched": bool according to threshold,
            "all_scores": {person_id: best similarity for that person},
        }
    """

    all_scores: dict[str, float] = {}
    best_person = None
    best_similarity = -1.0

    for person_id, embeddings in reference_database.items():
        if not embeddings:
            continue
        person_similarity = max(
            cosine_similarity(query_embedding, reference) for reference in embeddings
        )
        all_scores[person_id] = person_similarity
        if person_similarity > best_similarity:
            best_similarity = person_similarity
            best_person = person_id

    return {
        "person_id": best_person,
        "similarity": best_similarity if best_similarity >= 0 else 0.0,
        "matched": bool(best_similarity >= threshold),
        "all_scores": all_scores,
    }
