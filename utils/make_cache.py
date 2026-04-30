import hashlib

def make_cache_key(query: str, top_k: int) -> str:
    raw = f"{query.strip().lower()}:{top_k}"
    return hashlib.md5(raw.encode()).hexdigest()