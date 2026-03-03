# evaluation/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Systematically compare different RAG configurations to determine
# which settings produce the best answers.
#
# This is the code behind your "Evaluation Report" portfolio deliverable.
# It tests 3 different chunking strategies and measures their impact.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Define the test configurations ────────────────────────────────────────────
# We test 3 chunking strategies — small, medium, large chunks
CHUNKING_CONFIGS = [
    {
        "name": "Small Chunks",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "description": "More precise but may lose context"
    },
    {
        "name": "Medium Chunks (Baseline)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "description": "Balanced — recommended default"
    },
    {
        "name": "Large Chunks",
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "description": "More context but may retrieve irrelevant text"
    }
]

# ── Test questions with expected answer keywords ───────────────────────────────
# In a real evaluation you'd use human judges. Here we use keyword matching
# as a proxy for correctness — good enough for a portfolio evaluation.
TEST_QUESTIONS = [
    {
        "question": "What is the penalty for plagiarism?",
        "expected_keywords": ["mark", "zero", "misconduct", "penalty", "expel"],
        "category": "Academic Integrity"
    },
    {
        "question": "How do I request an extension?",
        "expected_keywords": ["extension", "deadline", "submit", "request", "form"],
        "category": "Assessment"
    },
    {
        "question": "What disability support is available?",
        "expected_keywords": ["disability", "support", "adjustment", "accessible"],
        "category": "Student Support"
    }
]


def build_index_for_config(config: dict, pdf_folder: str = "data/pdfs") -> FAISS:
    """Builds a FAISS index with a specific chunking configuration."""

    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def score_answer(answer: str, expected_keywords: list) -> float:
    """
    Simple keyword-based scoring.
    Returns the proportion of expected keywords found in the answer.
    Score of 1.0 = all keywords present, 0.0 = none present.

    NOTE: In a real evaluation you'd use human raters or an LLM-as-judge approach.
    This is a simplified proxy for portfolio purposes.
    """
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def run_evaluation():
    """
    Runs the full evaluation across all configurations and test questions.
    Saves results to evaluation/results.json
    """

    print("🔬 Starting EduRAG Evaluation")
    print("=" * 60)

    # Build the LLM for generating answers
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
Answer this question based only on the context provided.
Context: {context}
Question: {question}
Answer:""")
    chain = prompt | llm | StrOutputParser()

    all_results = []

    for config in CHUNKING_CONFIGS:
        print(f"\n📐 Testing: {config['name']}")
        print(f"   Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}")

        # Build index for this configuration
        vectorstore, num_chunks = build_index_for_config(config)
        print(f"   Built index with {num_chunks} chunks")

        config_results = {
            "config": config,
            "num_chunks": num_chunks,
            "question_results": []
        }

        for test in TEST_QUESTIONS:
            q_start = time.time()

            # Retrieve relevant chunks
            docs = vectorstore.similarity_search(test["question"], k=4)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Generate answer
            answer = chain.invoke({"context": context, "question": test["question"]})

            # Score the answer
            score = score_answer(answer, test["expected_keywords"])
            response_time = time.time() - q_start

            result = {
                "question": test["question"],
                "category": test["category"],
                "answer_preview": answer[:200],
                "keyword_score": round(score, 2),
                "response_time_seconds": round(response_time, 2),
                "chunks_retrieved": len(docs)
            }

            config_results["question_results"].append(result)

            print(f"   ✓ Q: {test['question'][:50]}...")
            print(f"     Score: {score:.0%} | Time: {response_time:.1f}s")

        # Calculate average score for this configuration
        avg_score = sum(r["keyword_score"] for r in config_results["question_results"]) / len(TEST_QUESTIONS)
        config_results["average_score"] = round(avg_score, 2)

        print(f"\n   📊 Average Score: {avg_score:.0%}")
        all_results.append(config_results)

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Avg Score':<15} {'Num Chunks':<15}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['config']['name']:<30} {result['average_score']:.0%}{'':10} {result['num_chunks']:<15}")

    print(f"\n✅ Full results saved to evaluation/results.json")


if __name__ == "__main__":
    run_evaluation()