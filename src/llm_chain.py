# src/llm_chain.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Take the user's question and the retrieved document chunks,
# combine them into a well-crafted prompt, send it to the LLM,
# and return a grounded, cited answer.
#
# This is the "brain" of the system. Without the retrieved context,
# the LLM would answer from memory (and potentially hallucinate).
# WITH the context, it can only answer from what's in YOUR documents.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ── The Prompt Template ────────────────────────────────────────────────────────
# This is the most important part of a RAG system.
# We give the LLM clear instructions:
#   1. Only use the provided context (prevents hallucination)
#   2. Always cite sources (adds transparency)
#   3. Admit when it doesn't know (prevents confident wrong answers)
#   4. Keep answers appropriate for a university audience

RAG_PROMPT_TEMPLATE = """You are EduRAG, a helpful AI assistant for students and staff at a UK university.
Your job is to answer questions ONLY based on the document excerpts provided below.

STRICT RULES:
- Only use information from the PROVIDED CONTEXT below
- Always mention which document and page your answer comes from
- If the answer is not in the provided context, say: "I couldn't find information about this in the available documents. Please contact your institution directly."
- Do not make up policies, regulations, or procedures
- Be concise, clear and helpful
- If information seems contradictory across documents, mention both versions

PROVIDED CONTEXT FROM UNIVERSITY DOCUMENTS:
─────────────────────────────────────────
{context}
─────────────────────────────────────────

STUDENT/STAFF QUESTION: {question}

YOUR ANSWER (with source citations):"""


def build_rag_chain():
    """
    Builds and returns a LangChain RAG chain.

    A "chain" in LangChain is a pipeline:
    Prompt Template → LLM → Output Parser
    """

    # Initialise the LLM
    # gpt-4o-mini is cheap (~$0.00015/1K input tokens) and very capable
    # temperature=0 means deterministic/factual answers (no creativity)
    # temperature=1 would make it more creative/varied
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,                               # Factual, consistent answers
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000                              # Limit response length
    )

    # Create the prompt template
    # The {context} and {question} placeholders get filled in at runtime
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Build the chain using LangChain's pipe operator (|)
    # Think of | like a conveyor belt: input → prompt → llm → parse output
    chain = prompt | llm | StrOutputParser()

    return chain


def generate_answer(
    question: str,
    context: str,
    chain=None
) -> str:
    """
    Generates a grounded answer using the LLM chain.

    Parameters:
    -----------
    question : The user's question
    context  : The formatted retrieved chunks (from retriever.py)
    chain    : The LLM chain (built once and reused for efficiency)

    Returns:
    --------
    A string containing the LLM's answer with source citations
    """

    # Build chain if not provided (lazy initialisation)
    if chain is None:
        chain = build_rag_chain()

    # Run the chain — this sends the API request to OpenAI
    # The chain fills in {context} and {question} in the prompt template
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response


def generate_answer_streaming(
    question: str,
    context: str,
    chain=None
):
    """
    Streaming version — yields the answer token by token.
    This makes the UI feel faster because text appears as it's generated,
    rather than waiting for the full response.

    Used with Streamlit's st.write_stream()
    """

    if chain is None:
        chain = build_rag_chain()

    # .stream() returns a generator that yields chunks of text
    for chunk in chain.stream({
        "context": context,
        "question": question
    }):
        yield chunk


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_context = """
[Source 1: student_handbook.pdf, Page 12]
Academic misconduct includes plagiarism, collusion, and fabrication of data.
Students found guilty of academic misconduct may receive a mark of zero
for the affected assessment, or in serious cases, be expelled from the university.

[Source 2: assessment_regulations.pdf, Page 5]
All suspected cases of academic misconduct must be reported to the
Academic Integrity Officer within 10 working days of discovery.
"""

    test_question = "What happens if a student is caught plagiarising?"

    chain = build_rag_chain()
    answer = generate_answer(test_question, test_context, chain)

    print(f"Q: {test_question}")
    print(f"\nA: {answer}")