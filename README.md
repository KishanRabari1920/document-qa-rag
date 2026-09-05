# Document QA RAG — Three Implementations (OpenAI, Ollama, Hybrid Search)

A Retrieval-Augmented Generation (RAG) project that lets you ask natural-language questions over your own PDF/TXT/CSV documents and get grounded answers with source citations. This repo contains **three progressively advanced versions** of the same core idea, so you can compare a cloud-based setup against a fully local, free setup, and a hybrid-retrieval upgrade.

## What's inside

| File | Approach | Embeddings | LLM | Retrieval |
|---|---|---|---|---|
| `openai_rag.py` | Cloud-based | OpenAI `text-embedding-3-small` | OpenAI `gpt-4o-mini` | Dense vector search (FAISS) |
| `ollama_rag.py` | Fully local & free | HuggingFace `all-MiniLM-L6-v2` | Ollama `llama3.2` | Dense vector search (FAISS), multi-file loader (PDF/TXT/CSV) |
| `advance_rag.py` | Advanced local | HuggingFace `all-MiniLM-L6-v2` | Ollama `llama3.2` | **Hybrid**: Dense (FAISS) + Sparse (BM25), deduplicated |

All three follow the same pipeline:
1. Load document(s)
2. Split into overlapping chunks
3. Embed chunks
4. Store in a FAISS vector database
5. Retrieve relevant chunks for a user question
6. Build a grounded prompt (answer-only-from-context, refuse if not found)
7. Generate an answer with the LLM
8. Interactive Q&A loop in the terminal until `exit`

## Why three versions?

- **`openai_rag.py`** — the simplest baseline, good when you want top-tier answer quality and don't mind API costs.
- **`ollama_rag.py`** — a zero-cost, fully offline alternative using a local LLM (Ollama) and free local embeddings. Supports loading a whole folder of mixed PDF/TXT/CSV files.
- **`advance_rag.py`** — adds **hybrid retrieval** (combining dense semantic search with BM25 keyword search) to improve recall on queries that rely on exact terms/numbers that embeddings alone can miss, plus chunk-level metadata (source, page, chunk ID) for citation in answers.

## Project structure

```
.
├── advance_rag.py          # Hybrid (FAISS + BM25) local RAG
├── ollama_rag.py            # Local RAG (FAISS only, multi-file loader)
├── openai_rag.py             # Cloud RAG using OpenAI
├── data/                      # Put your PDF / TXT / CSV files here (used by ollama_rag.py)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> Note: `advance_rag.py` currently points at a single hardcoded PDF path — update the `pdf_path` variable to point at your own document, or adapt it to loop over the `data/` folder like `ollama_rag.py` does.

## Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. For the local versions (`ollama_rag.py`, `advance_rag.py`)
Install [Ollama](https://ollama.com/) and pull the model:
```bash
ollama pull llama3.2
```

### 4. For the OpenAI version (`openai_rag.py`)
Copy `.env.example` to `.env` and add your key:
```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

### 5. Add your documents
Place your PDF/TXT/CSV files inside the `data/` folder (for `ollama_rag.py`), or update the file path variable in the script you're running.

## Usage

```bash
# Cloud version (OpenAI)
python openai_rag.py

# Local version (Ollama, FAISS only)
python ollama_rag.py

# Local version with hybrid search (FAISS + BM25)
python advance_rag.py
```

Then type your question at the prompt. Type `exit` to quit.

## Example

```
Ask your question: What is supervised learning?

ANSWER:
Supervised learning is a type of machine learning where the model is trained
on labeled data... (Source: Unit 1_Introduction to Machine Learning.pdf, Page 3)
```

## Tech stack

- [LangChain](https://www.langchain.com/) — document loaders, text splitting, retrievers
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) (via LangChain's `BM25Retriever`) — sparse keyword retrieval
- [HuggingFace Sentence Transformers](https://www.sbert.net/) — free local embeddings
- [Ollama](https://ollama.com/) — local LLM inference
- [OpenAI API](https://platform.openai.com/) — cloud LLM + embeddings (optional)

## Roadmap / possible improvements

- [ ] Add re-ranking (cross-encoder) after hybrid retrieval
- [ ] Add a simple web UI (Streamlit/FastAPI) instead of terminal input
- [ ] Persist FAISS index to disk so it doesn't rebuild every run
- [ ] Add automated evaluation (answer relevance, faithfulness)
- [ ] Unify all three scripts behind one CLI with a `--mode` flag

## License

MIT
