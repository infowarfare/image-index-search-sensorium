import os
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
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


# --- Endpoints ---
@app.get("/")
def root():
    return FileResponse("static/search_index.html")


@app.post("/index", status_code=202)
async def index_images(request: Request, files: List[UploadFile] = File(...)):
    """Save uploaded images to disk and index them via CLIP."""
    logger.info(f"Indexierung gestartet – {len(files)} Bilder")
    saved_paths = []
    for file in files:
        dest = IMAGE_DIR / file.filename
        dest.write_bytes(await file.read())
        saved_paths.append(str(dest.resolve()))

    # ImageFileToDocument expects file paths – same as your notebook
    await request.app.state.indexing_pipeline.run_async(
        {"image_converter": {"sources": saved_paths}}
    )
    logger.info(f"Indexierung abgeschlossen – {len(saved_paths)} Bilder erfolgreich indexiert")
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
        filename = Path(str(doc.meta["file_path"])).name  # nur "apple.jpg"
        file_path = IMAGE_DIR / filename                   # "images/apple.jpg"
        if not file_path.exists():
            logger.warning(f"Datei nicht gefunden: {file_path}")
            print(f"Nicht gefunden: {file_path}")
            continue
        image_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        results.append(SearchResult(
            filename=filename,
            score=doc.score,
            image_base64=image_b64,
        ))

    if not results:
        logger.warning(f"Keine Ergebnisse für Query: '{body.query}'")
        raise HTTPException(status_code=404, detail="Dateien nicht lokal vorhanden.")
    
    logger.info(f"Suche erfolgreich – {len(results)} Ergebnisse zurückgegeben")
    return results


@app.get("/stats") # TODO: replace this with /health
async def stats(request: Request):
    count = request.app.state.doc_store.count_documents()
    return {"indexed_images": count}