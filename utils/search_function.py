import httpx
from haystack.tools import Tool
from haystack.dataclasses import ChatMessage, ImageContent, TextContent
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from resize_image import resize_image

def search_images(query: str, top_k: int = 3) -> list[dict]:
    """Searches for images in the database using a natural language query."""
    with httpx.Client() as client:
        res = client.post(
            "http://localhost:8000/search",
            json={"query": query, "top_k": top_k},
            timeout=30.0
        )
        res.raise_for_status()
        docs = res.json()

    content = []
    for doc in docs:
        print(f"base64 length: {len(doc['image_base64'])}")
        meta_parts = [f"Dateiname: {doc['filename']} | Score: {round(doc['score'], 4)}"]
        if doc.get("description"):
            meta_parts.append(f"Beschreibung: {doc['description']}")
        if doc.get("keywords"):
            meta_parts.append(f"Keywords: {', '.join(doc['keywords'])}")
        content.append(TextContent(text=" | ".join(meta_parts)))

        resized_b64 = resize_image(doc["image_base64"])
        print(f"resized base64 length: {len(resized_b64)}")
        content.append(ImageContent(
            base64_image=resized_b64,
            mime_type="image/jpeg"
        ))

    return content