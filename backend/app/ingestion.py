import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.embeddings import embedding_service
from backend.app.vector_store import vector_store


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[str]:
    """Splits a document text into overlapping chunks using paragraph and sentence boundaries.

    Args:
        text: The raw document text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    # Split into paragraphs first
    paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]
    
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{paragraph}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from the end of current_chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + "\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Single paragraph is larger than chunk_size, split by sliding window
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end].strip())
                    start += chunk_size - chunk_overlap
                current_chunk = ""

    if current_chunk and current_chunk not in chunks:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def load_and_chunk_documents(
    documents_dir: Path = settings.DOCUMENTS_PATH
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Reads all .txt files from the documents directory and breaks them into chunks.

    Args:
        documents_dir: Path to directory containing .txt files.

    Returns:
        Tuple containing:
        - chunk_ids: List of unique string IDs
        - chunk_metadatas: List of metadata dicts
        - chunk_documents: List of text chunk strings
    """
    if not documents_dir.exists():
        logger.warning(f"Documents directory does not exist: {documents_dir}")
        return [], [], []

    txt_files = list(documents_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt documents found in {documents_dir}")
        return [], [], []

    logger.info(f"Found {len(txt_files)} .txt document(s) in {documents_dir}")

    all_ids: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_documents: List[str] = []

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunk_text(content)
            logger.info(f"Loaded '{file_path.name}': created {len(chunks)} chunk(s)")

            for idx, chunk in enumerate(chunks):
                # Deterministic chunk ID based on filename and chunk index
                file_hash = hashlib.md5(file_path.name.encode()).hexdigest()[:8]
                chunk_id = f"{file_path.stem}_{file_hash}_{idx:03d}"

                metadata = {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }

                all_ids.append(chunk_id)
                all_metadatas.append(metadata)
                all_documents.append(chunk)

        except Exception as e:
            logger.error(f"Failed to read and process document '{file_path}': {e}")

    return all_ids, all_metadatas, all_documents


def ingest_documents(
    documents_dir: Path = settings.DOCUMENTS_PATH,
    force_reload: bool = False
) -> Dict[str, Any]:
    """Ingests all .txt files from the target directory into the persistent ChromaDB vector store.

    Args:
        documents_dir: Path to directory containing .txt files.
        force_reload: If True, clears existing collection and re-indexes.

    Returns:
        Dictionary summary of ingestion results.
    """
    if force_reload:
        logger.info("Force reload requested. Clearing existing vector store collection...")
        vector_store.reset_collection()

    ids, metadatas, documents = load_and_chunk_documents(documents_dir)

    if not ids:
        return {
            "status": "warning",
            "message": "No document chunks to ingest.",
            "files_processed": [],
            "total_chunks": vector_store.count()
        }

    logger.info(f"Generating embeddings for {len(documents)} document chunk(s)...")
    embeddings = embedding_service.get_embeddings(documents)

    logger.info("Storing chunk embeddings into persistent ChromaDB...")
    vector_store.add_chunks(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    processed_files = sorted(list(set(m["source"] for m in metadatas)))
    total_count = vector_store.count()

    logger.info(f"Ingestion complete. Total indexed chunks: {total_count}")

    return {
        "status": "success",
        "message": f"Successfully ingested {len(ids)} chunks from {len(processed_files)} file(s).",
        "files_processed": processed_files,
        "total_chunks": total_count
    }


if __name__ == "__main__":
    # Allow running ingestion standalone: python -m backend.app.ingestion
    result = ingest_documents(force_reload=True)
    print(result)
