import json
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipelines import build_document_store, build_indexing_pipeline, build_search_pipeline

from dotenv import load_dotenv
load_dotenv()

from logger import logger

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to Qdrant and building pipelines...")
    doc_store = build_document_store()
    app.state.doc_store = doc_store
    app.state.indexing_pipeline = build_indexing_pipeline(doc_store)
    app.state.search_pipeline = build_search_pipeline(doc_store)
    app.state.indexing_pipeline.warm_up()
    app.state.search_pipeline.warm_up()
    print("Ready.")
    yield


app = FastAPI(title="CLIP Image Search API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    filename: str
    score: float | None
    image_base64: str
    description: str | None = None
    keywords: list[str] = []


# --- Endpoints ---
@app.get("/home")
def root():
    return FileResponse("static/search_index.html")


@app.post("/index", status_code=202)
async def index_images(
    request: Request,
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
):
    """Save uploaded images to disk and index them via CLIP.

    Optionally accepts a JSON string via the `metadata` form field.
    Expected shape:
        {
            "<filename>": {
                "description": "...",
                "keywords": ["tag1", "tag2"]
            },
            ...
        }
    """
    logger.info(f"Indexierung gestartet – {len(files)} Bilder")

    # Parse optional metadata map keyed by filename
    meta_map: dict[str, dict] = {}
    if metadata:
        try:
            meta_map = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning("Metadata JSON konnte nicht geparst werden – wird ignoriert.")

    saved_paths = []
    for file in files:
        dest = IMAGE_DIR / file.filename
        dest.write_bytes(await file.read())
        saved_paths.append(str(dest))
        #saved_paths.append(str(dest.resolve()))

    # Step 1: convert images to Documents
    converter_result = request.app.state.indexing_pipeline.get_component(
        "image_converter"
    ).run(sources=saved_paths)
    documents = converter_result["documents"]

    # Step 2: inject metadata into each Document
    for doc in documents:
        filename = Path(str(doc.meta.get("file_path", ""))).name
        extra = meta_map.get(filename, {})
        if extra.get("description"):
            doc.meta["description"] = extra["description"]
        if extra.get("keywords"):
            kw = extra["keywords"]
            # Store as list; also append to content so CLIP text context is enriched
            doc.meta["keywords"] = kw
            # Optionally surface keywords as part of the document's text content
            tag_string = ", ".join(kw)
            doc.content = (
                f"{extra.get('description', '')} {tag_string}".strip()
                if extra.get("description")
                else tag_string
            ) or doc.content

    # Step 3: embed + write
    embed_result = request.app.state.indexing_pipeline.get_component(
        "image_doc_embedder"
    ).run(documents=documents)
    embedded_docs = embed_result["documents"]

    request.app.state.indexing_pipeline.get_component(
        "document_writer"
    ).run(documents=embedded_docs)

    logger.info(
        f"Indexierung abgeschlossen – {len(saved_paths)} Bilder erfolgreich indexiert"
    )
    return {"message": f"{len(saved_paths)} images indexed.", "files": saved_paths}


@app.post("/search", response_model=List[SearchResult])
async def search(request: Request, body: SearchRequest):
    logger.info(f"Suchanfrage: '{body.query}' | top_k={body.top_k}")
    result = await request.app.state.search_pipeline.run_async(
        {"text_embedder": {"text": body.query}}
    )

    docs = result["retriever"]["documents"][: body.top_k]
    for doc in docs:
        logger.debug(f"  Score: {doc.score:.4f} | Pfad: {doc.meta.get('file_path')}")

    logger.debug(f"Gefundene Dokumente: {len(docs)}")

    if not docs:
        raise HTTPException(status_code=404, detail="No matching images found.")

    results = []
    for doc in docs:
        filename = Path(str(doc.meta["file_path"])).name
        file_path = IMAGE_DIR / filename
        if not file_path.exists():
            logger.warning(f"Datei nicht gefunden: {file_path}")
            continue
        image_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        results.append(SearchResult(
            filename=filename,
            score=doc.score,
            image_base64=image_b64,
            description=doc.meta.get("description") or None,
            keywords=doc.meta.get("keywords") or [],
        ))

    if not results:
        logger.warning(f"Keine Ergebnisse für Query: '{body.query}'")
        raise HTTPException(status_code=404, detail="Dateien nicht lokal vorhanden.")

    logger.info(f"Suche erfolgreich – {len(results)} Ergebnisse zurückgegeben")
    return results


@app.get("/stats")
async def stats(request: Request):
    count = request.app.state.doc_store.count_documents()
    return {"indexed_images": count}