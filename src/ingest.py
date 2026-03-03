# src/ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Read all PDFs in data/pdfs/, split into chunks, convert to vector
# embeddings, and save a searchable FAISS index to data/faiss_index/.
#
# FIX APPLIED: OpenAIEmbeddings is instantiated INSIDE ingest_documents()
# rather than at module level, avoiding the httpx 'proxies' conflict that
# occurs when the object is created on import in newer environments.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def ingest_documents(
    pdf_folder: str = "data/pdfs",
    index_save_path: str = "data/faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Reads all PDFs in pdf_folder, creates embeddings, and saves a FAISS index.

    Parameters:
    -----------
    pdf_folder      : Where your PDF files live
    index_save_path : Where to save the finished FAISS database
    chunk_size      : How many characters per chunk (~150 words at 1000 chars)
    chunk_overlap   : Overlap between chunks to preserve context at boundaries
    """

    print(f"📂 Loading PDFs from: {pdf_folder}")

    # ── STEP 1: Load all PDFs ──────────────────────────────────────────────────
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()

    if not documents:
        print("❌ No documents found. Check that PDF files exist in:", pdf_folder)
        return None

    print(f"✅ Loaded {len(documents)} pages from PDFs")

    # ── STEP 2: Split into chunks ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️  Split into {len(chunks)} chunks")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    # Show a sample so you can verify the split looks sensible
    print("\n📄 Sample chunk:")
    print("-" * 50)
    print(chunks[0].page_content[:300])
    print(f"\nMetadata: {chunks[0].metadata}")
    print("-" * 50)

    # ── STEP 3: Create embeddings ──────────────────────────────────────────────
    # KEY FIX: OpenAIEmbeddings instantiated HERE, inside the function.
    # Previously it was at module level, which caused the httpx 'proxies'
    # TypeError when the module was imported in environments with httpx>=0.28.
    print("\n🔢 Creating embeddings (this may take a minute)...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # ── STEP 4: Build and save the FAISS index ─────────────────────────────────
    print("🗄️  Building FAISS vector store...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(index_save_path, exist_ok=True)
    vectorstore.save_local(index_save_path)

    print(f"\n🎉 Done! FAISS index saved to: {index_save_path}")
    print(f"   Total vectors stored: {vectorstore.index.ntotal}")

    return vectorstore


if __name__ == "__main__":
    ingest_documents()