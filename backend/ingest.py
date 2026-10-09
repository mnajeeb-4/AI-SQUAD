import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


# Configuration
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file")

# Embedding model produces 768-dimensional vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Chunking settings
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks to preserve context

# Documents to ingest
DOCUMENTS = [
    {
        "title": "First Aid Reference Guide",
        "source": "medical",
        "path": "first_aid.pdf",
    },
    {
        "title": "Pakistan Law Code",
        "source": "legal",
        "path": "pakistan_law.pdf",
    },
]


# Clients

print("Connecting to Supabase...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print("Loading embedding model (this may take a moment the first time)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# Helper Functions

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Read a PDF file and return all of its text content as a single string.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(text)
        print(f"  - Extracted page {page_number}/{len(reader.pages)}")

    full_text = "\n".join(pages)
    return full_text


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace and normalize the text
    so that chunks are consistent and clean.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long text into overlapping chunks of approximately `chunk_size`
    characters. Overlap helps preserve context across chunk boundaries.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if len(chunk) > 50:  # skip tiny fragments
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def generate_embedding(text: str) -> list[float]:
    """
    Convert a piece of text into a 768-dimensional vector.
    """
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def create_document_record(title: str, source: str) -> str:
    """
    Insert a new row in the `documents` table and return its ID.
    """
    response = (
        supabase.table("documents")
        .insert({"title": title, "source": source})
        .execute()
    )
    return response.data[0]["id"]


def store_chunks(document_id: str, chunks: list[str]) -> None:
    """
    Generate embeddings for each chunk and insert them into the
    `chunks` table in batches.
    """
    batch_size = 50
    total = len(chunks)

    for index in range(0, total, batch_size):
        batch = chunks[index : index + batch_size]
        rows = []

        for chunk in batch:
            embedding = generate_embedding(chunk)
            rows.append(
                {
                    "document_id": document_id,
                    "content": chunk,
                    "embedding": embedding,
                }
            )

        supabase.table("chunks").insert(rows).execute()
        print(f"  - Inserted chunks {index + 1} to {min(index + batch_size, total)} of {total}")


# Main Ingestion Flow

def ingest_document(doc: dict) -> None:
    """
    Process a single PDF: extract text, clean it, split it,
    and store its chunks in Supabase.
    """
    pdf_path = Path(__file__).parent / doc["path"]

    if not pdf_path.exists():
        print(f"[SKIP] File not found: {pdf_path}")
        return

    print(f"\nProcessing: {doc['title']}")
    print(f"  File: {pdf_path}")

    raw_text = extract_text_from_pdf(str(pdf_path))
    cleaned_text = clean_text(raw_text)

    print(f"  Total characters extracted: {len(cleaned_text)}")

    chunks = split_into_chunks(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Total chunks created: {len(chunks)}")

    document_id = create_document_record(doc["title"], doc["source"])
    print(f"  Document record created with ID: {document_id}")

    store_chunks(document_id, chunks)
    print(f"  Done ingesting '{doc['title']}'")


def main() -> None:
    print("Starting RAG ingestion")

    for doc in DOCUMENTS:
        ingest_document(doc)

    
    print("RAG ingestion completed successfully")


if __name__ == "__main__":
    main()