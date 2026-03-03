# src/retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Load the pre-built FAISS index and provide a function that takes a
# question and returns the most relevant document chunks.
#
# Think of this as the "librarian" — you ask a question,
# they fetch the most relevant paragraphs from the indexed documents.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_vectorstore(index_path: str = "data/faiss_index") -> FAISS:
    """
    Loads the saved FAISS index from disk.
    We must pass the same embedding model used during ingestion,
    so the query gets converted to vectors in the same "space".
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # allow_dangerous_deserialization=True is required for FAISS with LangChain
    # It's safe here because WE created this index ourselves
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"✅ Loaded FAISS index with {vectorstore.index.ntotal} vectors")
    return vectorstore


def retrieve_relevant_chunks(
    question: str,
    vectorstore: FAISS,
    top_k: int = 4
) -> list:
    """
    Takes a user question and returns the top_k most relevant document chunks.

    How it works:
    1. The question is converted to a vector embedding (same model as ingestion)
    2. FAISS compares this vector against ALL stored vectors using cosine similarity
    3. The top_k closest matches are returned

    Parameters:
    -----------
    question    : The user's natural language question
    vectorstore : The loaded FAISS index
    top_k       : How many chunks to retrieve (4 is usually a good balance)
                  Too few = missing context | Too many = LLM gets confused

    Returns:
    --------
    List of Document objects, each with .page_content and .metadata
    """

    # similarity_search does the vector comparison
    # It returns Document objects sorted by relevance (most relevant first)
    docs = vectorstore.similarity_search(
        query=question,
        k=top_k
    )

    return docs


def format_retrieved_chunks(docs: list) -> str:
    """
    Formats retrieved chunks into a clean string for the LLM prompt.
    Also extracts source information for citations.

    Each chunk is labelled with its source file and page number so the
    LLM can reference them in its answer.
    """

    formatted_chunks = []

    for i, doc in enumerate(docs, start=1):
        # Extract source metadata
        source_file = doc.metadata.get("source", "Unknown source")
        page_num = doc.metadata.get("page", "Unknown page")

        # Clean up the file path to just show the filename
        source_file = os.path.basename(source_file)

        chunk_text = f"""
[Source {i}: {source_file}, Page {page_num}]
{doc.page_content}
"""
        formatted_chunks.append(chunk_text)

    return "\n---\n".join(formatted_chunks)


def get_sources_list(docs: list) -> list:
    """
    Returns a clean list of source citations for displaying in the UI.
    """
    sources = []
    for doc in docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", "?")
        citation = f"📄 {source_file} — Page {page_num}"

        # Avoid duplicate citations
        if citation not in sources:
            sources.append(citation)

    return sources


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vs = load_vectorstore()

    test_question = "What is the policy on academic misconduct?"
    docs = retrieve_relevant_chunks(test_question, vs)

    print(f"\n🔍 Question: {test_question}")
    print(f"📚 Retrieved {len(docs)} chunks:\n")

    for i, doc in enumerate(docs, 1):
        print(f"Chunk {i} — {doc.metadata}")
        print(doc.page_content[:200])
        print()