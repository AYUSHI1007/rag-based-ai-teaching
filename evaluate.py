import os
import sys
import json
import argparse
import joblib
import numpy as np
from rag_strategies import RAGPipeline


def load_dataset(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_query_embeddings(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_corpus(joblib_path):
    df = joblib.load(joblib_path)
    chunks = []
    embeddings = []

    for lec, group in df.groupby("number", sort=False):
        for i, (_, row) in enumerate(group.iterrows()):
            chunks.append({
                "id": f"{lec}_{i}",
                "lecture": str(lec).zfill(2),
                "text": row["text"],
                "start": row.get("start", 0),
                "end": row.get("end", 0)
            })
            embeddings.append(row["embedding"])

    return chunks, np.array(embeddings)


def calculate_hit(expected_ids, retrieved_ids):
    expected_set = set(expected_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def run_evaluation(pipeline, dataset, query_embeddings, top_k=3):
    strategies = [
        ("Simple Chunking", lambda q, emb: pipeline.retrieve_simple(emb, top_k=top_k)),
        ("Semantic Chunking", lambda q, emb: pipeline.retrieve_semantic(emb, top_k=top_k)),
        ("Hybrid Search", lambda q, emb: pipeline.retrieve_hybrid(q, emb, top_k=top_k)),
        ("Reranking", lambda q, emb: pipeline.retrieve_rerank(q, emb, top_k=top_k))
    ]

    results = {}
    total_questions = len(dataset)

    for name, retrieve_fn in strategies:
        total_hits = 0
        reciprocal_ranks = []

        for item in dataset:
            qid = str(item["id"])
            query = item["question"]
            expected_ids = item["expected_chunk_ids"]
            query_emb = query_embeddings.get(qid)

            retrieved = retrieve_fn(query, query_emb)
            hit, rr = calculate_hit(expected_ids, retrieved)
            total_hits += hit
            reciprocal_ranks.append(rr)

        accuracy = (total_hits / total_questions) * 100.0
        mrr = sum(reciprocal_ranks) / total_questions if total_questions else 0.0
        results[name] = {"accuracy": accuracy, "mrr": mrr}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--use_cross_encoder", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(base_dir, "evaluation", "eval_dataset.json")
    cache_path = os.path.join(base_dir, "evaluation", "query_embeddings_cache.json")
    joblib_path = os.path.join(base_dir, "embeddings.joblib")

    print("\nLoading corpus and embeddings...")
    chunks, embeddings = load_corpus(joblib_path)
    dataset = load_dataset(eval_path)
    query_embeddings = load_query_embeddings(cache_path)

    print(f"Loaded {len(chunks)} chunks and {len(dataset)} evaluation questions.")

    reranker = None
    if args.use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder
            print("Loading CrossEncoder (BAAI/bge-reranker-v2-m3)...")
            reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        except Exception as e:
            print("CrossEncoder load skipped:", e)

    pipeline = RAGPipeline(chunks=chunks, embeddings=embeddings, reranker_model=reranker)

    print("\nRunning Evaluation across 4 Strategies (50 Qs x 4 Methods)...")
    results = run_evaluation(pipeline, dataset, query_embeddings, top_k=args.top_k)

    print("\n" + "=" * 65)
    print(f"{'Strategy':<25} | {'Accuracy (Hit Rate@' + str(args.top_k) + ')':<22} | {'MRR':<10}")
    print("=" * 65)

    for strategy, metrics in results.items():
        print(f"{strategy:<25} | {metrics['accuracy']:>6.1f}%{'':<15} | {metrics['mrr']:>6.3f}")
    print("=" * 65)

    final_accuracy = results["Reranking"]["accuracy"] / 100.0
    print(f"\nFinal Reranking Accuracy: {final_accuracy * 100:.1f}% (Threshold: {args.threshold * 100:.1f}%)")

    if final_accuracy >= args.threshold:
        print("[SUCCESS] Build Passed: Accuracy meets or exceeds threshold.\n")
        sys.exit(0)
    else:
        print("[FAILED] Build Failed: Accuracy dropped below threshold.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
