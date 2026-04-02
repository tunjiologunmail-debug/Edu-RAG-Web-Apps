# 🎓 EduRAG — University Policy Q&A Assistant

![EduRAG Hero](edurag.png)

A Retrieval-Augmented Generation (RAG) system that allows students and staff
to query university policy documents in natural language, with cited sources.

Built as a portfolio project demonstrating applied AI skills for higher education.

## � Live Demo
Try the app here: https://edu-rag-web-apps-k85grzkutudyubt9o7y5cz.streamlit.app/

## �🏗️ Architecture
```
PDF Documents → Chunking → Embeddings → FAISS Index
                                              ↓
User Question → Embedding → Similarity Search → Top K Chunks
                                              ↓
                              Prompt Template + LLM → Cited Answer
```

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/yourusername/edurag.git
cd edurag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add PDF documents to data/pdfs/

# 6. Build the document index
python src/ingest.py

# 7. Launch the app
streamlit run app.py
```

## 📊 Evaluation

Run the chunking strategy comparison:
```bash
python evaluation/evaluate.py
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (local) |
| Framework | LangChain |
| UI | Streamlit |
| PDF Parsing | PyPDF |

## ⚠️ Responsible AI Notes

- Answers are grounded in uploaded documents to minimise hallucination
- Sources are always cited so users can verify answers
- A disclaimer is shown reminding users to verify critical decisions
- See `evaluation/hallucination_audit.md` for a full bias and hallucination assessment