# Haystack Image Search API

Semantic image search using CLIP embeddings, Haystack, Qdrant and FastAPI.

## How it works

Images are indexed using a CLIP image encoder and stored in Qdrant as vectors. A text query is encoded using a matching CLIP text encoder and compared against the stored image vectors to find the most similar images. Search results not only return the image itself, but metadata such as the image description and keywords. Used as an external image service for the Sensorium+ museum chatbot.

Additionally, a single page website offers the possibility to bulk index images with additional metadata. The indexed images can be retrieved via the search function, which also serves as the search endpoint for the chatbot.

## Setup

Create a `.env` file in the project root:
```
QDRANT_API_KEY=your-api-key
HF_TOKEN=your-api-key
TF_ENABLE_ONEDNN_OPTS=0
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Running

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running, then:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

To stop the container:
```bash
docker compose down
```

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/index` | Upload and index images |
| `POST` | `/search` | Search images with a text query |
| `GET` | `/stats` | Number of indexed images |

## Frontend

Open `static/search_index.html` in a browser to access the search and indexing UI.

## Models

| Task | Model |
|---|---|
| Image Embedding | `clip-ViT-B-32` |
| Text Embedding | `clip-ViT-B-32-multilingual-v1` |