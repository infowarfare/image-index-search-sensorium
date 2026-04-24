from dotenv import load_dotenv

load_dotenv()

import os
from haystack import AsyncPipeline
from haystack.utils import Secret
from haystack.components.converters.image import ImageFileToDocument
from haystack.components.embedders.image import SentenceTransformersDocumentImageEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever



def build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url="https://2cc4679a-ac81-4e79-bef8-0db94b84a037.eu-central-1-0.aws.cloud.qdrant.io",
        index="clip",
        api_key=Secret.from_env_var("QDRANT_API_KEY"),
        embedding_dim=512,
    )

def build_indexing_pipeline(doc_store: QdrantDocumentStore) -> AsyncPipeline:
    pipeline = AsyncPipeline()
    pipeline.add_component("image_converter", ImageFileToDocument(store_full_path=True))
    pipeline.add_component(
        "image_doc_embedder",
        SentenceTransformersDocumentImageEmbedder(model="clip-ViT-B-32", progress_bar=False)
    )
    pipeline.add_component("document_writer", DocumentWriter(document_store=doc_store))
    pipeline.connect("image_converter.documents", "image_doc_embedder.documents")
    pipeline.connect("image_doc_embedder.documents", "document_writer.documents")
    return pipeline

def build_search_pipeline(doc_store: QdrantDocumentStore) -> AsyncPipeline:
    pipeline = AsyncPipeline()
    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(
            model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
            progress_bar=False
        )
    )
    pipeline.add_component(
        "retriever",
        QdrantEmbeddingRetriever(document_store=doc_store, top_k=5)
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline