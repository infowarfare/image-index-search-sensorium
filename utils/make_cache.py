import hashlib

def make_cache_key(session_id: str, query: str, top_k: int) -> str:
    raw = f"{session_id}:{query.strip().lower()}:{top_k}"
    return hashlib.md5(raw.encode()).hexdigest()