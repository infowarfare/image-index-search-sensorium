# Haystack Image Search API

Semantic image search using CLIP embeddings, Haystack, Qdrant and FastAPI.

## How it works

Images are indexed using a CLIP image encoder and stored in Qdrant as vectors. A text query is encoded using a matching CLIP text encoder and compared against the stored image vectors to find the most similar images. Search results not only return the image itself, but metadata such as the image description and keywords. Used as an external image service for the Sensorium+ museum chatbot.

Addtionally, a single page website offers the possibility bulk index images with additional metadata. The indexed images can be retrieved via the search function provided that also serves as the search endpoint for the chatbot.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
QDRANT_API_KEY=your-api-key
HF_TOKEN =your-api-key
```

And suppress unnecessary warnings:
```
TF_ENABLE_ONEDNN_OPTS=0
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Running

```bash
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/index` | Upload and index images |
| `POST` | `/search` | Search images with a text query |
| `GET` | `/stats` | Number of indexed images |

## Frontend

Open `search_index.html` in a browser to access the search and indexing UI.

## Models

| Task | Model |
|---|---|
| Image Embedding | `clip-ViT-B-32` |
| Text Embedding | `clip-ViT-B-32-multilingual-v1` |