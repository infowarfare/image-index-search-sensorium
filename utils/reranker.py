from rapidfuzz import fuzz
from typing import Optional


def fuzzy_rerank(docs: list, query: str) -> list:
    """Rerankt Dokumente anhand von Fuzzy Matching der Query gegen die Metadaten."""
    scored = []
    for doc in docs:
        meta_score = _fuzzy_score(query, doc.meta)
        # Kombinierter Score aus CLIP + Fuzzy
        combined_score = (doc.score or 0.0) + meta_score
        doc.score = combined_score  # ← Score auf Dokument überschreiben
        scored.append((doc, combined_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored]


def _fuzzy_score(query: str, meta: dict) -> float:
    """Berechnet Fuzzy Score zwischen Query und Metadaten."""
    query = query.lower().strip()
    scores = []

    if description := meta.get("description"):
        scores.append(fuzz.partial_ratio(query, description.lower()) / 100)

    if keywords := meta.get("keywords"):
        if isinstance(keywords, list):
            keyword_scores = [
                fuzz.partial_ratio(query, kw.lower()) / 100
                for kw in keywords
            ]
            scores.append(max(keyword_scores))  # bestes Keyword gewinnt

    return sum(scores) / len(scores) if scores else 0.0