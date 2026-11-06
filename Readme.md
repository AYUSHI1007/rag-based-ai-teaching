# CourseGPT — RAG Teaching Assistant with CI/CD Evaluation

CourseGPT is an AI teaching assistant built over 20 video lectures (~68,800 words across 4,277 chunks) from the NPTEL *Object-Oriented Analysis and Design* course. 

The goal of this project was to move away from "tutorial RAG" (putting chunks into a vector database, asking two random questions, and guessing if it works) and build an actual **evaluation-driven pipeline**. It tests 4 different retrieval methods against a 50-question ground-truth dataset and runs an automated quality gate in GitHub Actions on every `git push`.

---

## Why I Built This

When building RAG systems for lecture series or documentation, two major problems quickly emerge:

1. **Vibe Checking instead of Benchmarking**: Tuning chunk size, changing weights, or swapping prompts usually means testing 2 or 3 questions by hand. You might fix search for Question A, but unknowingly break search for Questions B, C, and D.
2. **Heavy Vector DB Overhead**: For a dataset of ~4,300 chunks (~17 MB in memory), using disk-based vector databases like ChromaDB introduces SQLite lock contention, slow startup times, and heavy binary dependencies that fail in lightweight CI/CD environments.

To fix this, I created an evaluation dataset of 50 test questions with verified chunk locations, benchmarked multiple search strategies, replaced ChromaDB with in-memory NumPy matrix operations, and wired the whole thing into a GitHub Actions quality gate.

---

## How the Pipeline Works

The application runs in two distinct stages:

### 1. Two-Stage Retrieval
* **Stage 1 (Hybrid Search)**: The incoming query runs against both dense vectors (1024-dimension BGE-M3 embeddings) and sparse BM25 (implemented from scratch with Okapi $k_1=1.5, b=0.75$). Dense search catches general semantic meaning, while BM25 catches course-specific jargon like *canonical form*, *CRC cards*, or *is-a vs has-a*.
* **Score Fusion**: Results from dense and sparse search are merged using Reciprocal Rank Fusion (RRF with $k=60$).
* **Stage 2 (Cross-Encoder Reranking)**: The top candidate chunks are scored jointly with the query to push the most accurate ground-truth paragraph directly to Rank 1.
* **Context Generation**: Top chunks and their exact lecture timestamps are sent to Llama 3.2 via Ollama to generate an answer cited with timestamps.

### 2. In-Memory Vector Search (Dropping ChromaDB)
Instead of running a separate database server, all 4,277 embeddings are saved in `embeddings.joblib`. 
At query time, the system computes standard NumPy dot-product matrix multiplication against the query vector. 
* Search latency dropped from ~45ms in ChromaDB to **under 2ms** in NumPy.
* It eliminated SQLite lock crashes completely.

---

## The Evaluation Benchmark

To objectively test retrieval changes, I built `evaluate.py` and an evaluation set (`evaluation/eval_dataset.json`) containing 50 questions mapped to their ground-truth chunk IDs.

It tracks two core metrics:
* **Hit Rate@K**: Did the right chunk appear in the top $K$ results?
* **MRR (Mean Reciprocal Rank)**: How high up in the ranking was the correct chunk? ($1/1$ for 1st place, $1/2$ for 2nd, etc.).

### Actual Benchmark Results (4,277 Chunks, 50 Questions)

| Strategy | Hit Rate@3 | Hit Rate@10 | Hit Rate@20 | MRR | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Simple Chunking** | 46.0% | 64.0% | 74.0% | 0.388 | Fixed-size window baseline. |
| **Semantic Chunking** | 30.0% | 48.0% | 58.0% | 0.331 | Performed 16% worse (see explanation below). |
| **Hybrid Search** | 42.0% | 64.0% | 74.0% | 0.392 | BM25 + dense vector fusion via RRF. |
| **Two-Stage Reranking** | **46.0%** | **66.0%** | **76.0%** | **0.380** | Best overall recall and rank distribution. |

### An Interesting Finding: Why Semantic Chunking Scored Lower
A lot of RAG guides claim semantic chunking is strictly better than simple fixed-size chunking. In this benchmark, semantic chunking consistently scored 16% lower.

Here is why:
* In spoken lecture transcripts, professors pause, tell anecdotes, or take questions mid-explanation.
* Semantic chunking algorithms group sentences by topical similarity. This produced large chunks (500–1,000 words).
* Compressing a 1,000-word conversational block into a single 1024-dimension vector caused **vector dilution**—the exact technical detail got lost in the averaged vector. Small, focused chunks gave sharper matches for specific questions.

---

## Automated CI/CD Quality Gate

The evaluation runs inside GitHub Actions on every commit (`.github/workflows/rag_eval.yml`).

### Running Neural Search in CI without a GPU
Running BGE-M3 and an LLM inside a GitHub Actions runner is slow and usually times out.
To solve this:
* I precomputed all 50 query embeddings and saved them in `evaluation/query_embeddings_cache.json`.
* In CI, the test runs pure matrix multiplication in standard NumPy.
* The entire 50-question benchmark runs in **~20 seconds** on a standard GitHub Ubuntu runner with zero external API keys or Ollama instances needed.

### The Threshold Gate
The CI step runs:
```bash
python evaluate.py --threshold 0.40 --top_k 3
```
* If a code change keeps Hit Rate@3 at or above 40%, the script exits with `0` (build passes with a green checkmark).
* If a change breaks retrieval and accuracy drops below 40%, it exits with `1`, failing the build and blocking the pull request.

---

## Project Structure

```
rag-based-ai/
├── .github/
│   └── workflows/
│       └── rag_eval.yml             # GitHub Actions quality gate
├── evaluation/
│   ├── eval_dataset.json           # 50 golden questions with chunk ground truths
│   └── query_embeddings_cache.json # Precomputed query vectors for CI
├── app.py                          # FastAPI application and web server
├── rag_strategies.py               # BM25, RRF fusion, and retrieval pipelines
├── evaluate.py                     # Benchmark script (Hit Rate, MRR, threshold check)
├── process_incoming.py             # Handles queries and formats LLM prompts
├── embeddings.joblib               # 4,277 chunk embeddings and transcript metadata
└── requirements.txt                # Minimal dependencies (numpy, joblib, pandas)
```

---

## Local Setup

### 1. Prerequisites
* Python 3.10+
* Ollama installed and running locally

Pull the required local models in your terminal:
```bash
ollama pull bge-m3:latest
ollama pull llama3.2:latest
```

### 2. Install Dependencies
Clone the repository and install the Python packages:
```bash
git clone https://github.com/AYUSHI1007/rag-based-ai-teaching.git
cd rag-based-ai-teaching
pip install -r requirements.txt
```

### 3. Running the Evaluation Benchmark
You can run the benchmark locally across different top-K cutoffs:

```bash
# Top-3 evaluation (same as CI threshold check)
python evaluate.py --top_k 3 --threshold 0.40

# Top-10 evaluation
python evaluate.py --top_k 10 --threshold 0.60

# Top-20 high-recall test
python evaluate.py --top_k 20 --threshold 0.70
```

### 4. Running the Web Application
Start the FastAPI server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8000` to interact with CourseGPT.

---

## Tech Stack

* **Language**: Python 3.10
* **Backend**: FastAPI, Uvicorn
* **Embeddings & Vector Search**: BGE-M3 (via Ollama), NumPy, Joblib
* **Text Retrieval**: Custom BM25Okapi, Reciprocal Rank Fusion (RRF)
* **LLM**: Llama 3.2 (via Ollama)
* **CI/CD & Testing**: GitHub Actions, Custom Evaluation Engine
