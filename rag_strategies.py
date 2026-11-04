import math
import re
from collections import Counter
import numpy as np


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _initialize(self):
        df = Counter()
        for doc in self.corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for token in freq.keys():
                df[token] += 1

        total_docs = len(self.corpus)
        for token, freq in df.items():
            self.idf[token] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query_tokens, index):
        score = 0.0
        doc_len = self.doc_len[index]
        freq = self.doc_freqs[index]

        for token in query_tokens:
            if token not in freq:
                continue
            tf = freq[token]
            idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
        return score

    def search(self, query, top_k=10):
        query_tokens = self._tokenize(query)
        scores = [(i, self.score(query_tokens, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(dense_ranked_ids, sparse_ranked_ids, k=60):
    rrf_scores = {}

    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc_id in enumerate(sparse_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_results]


class RAGPipeline:
    def __init__(self, chunks, embeddings=None, reranker_model=None):
        self.chunks = chunks
        self.id_to_index = {chunk["id"]: i for i, chunk in enumerate(chunks)}
        self.embeddings = np.array(embeddings) if embeddings is not None else None

        tokenized_corpus = [re.findall(r"\w+", chunk["text"].lower()) for chunk in chunks]
        self.bm25 = BM25(tokenized_corpus)
        self.reranker = reranker_model

    def _dense_search(self, query_embedding, top_k=20):
        if self.embeddings is None:
            return []
        query_vec = np.array(query_embedding).reshape(1, -1)
        sims = np.dot(self.embeddings, query_vec.T).flatten()
        norm_product = (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)) + 1e-10
        cosine_sims = sims / norm_product
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]
        return [self.chunks[i]["id"] for i in top_indices]

    def _sparse_search(self, query, top_k=20):
        results = self.bm25.search(query, top_k=top_k)
        return [self.chunks[idx]["id"] for idx, _ in results]

    def retrieve_simple(self, query_embedding, top_k=5):
        return self._dense_search(query_embedding, top_k=top_k)

    def retrieve_semantic(self, query_embedding, top_k=5):
        initial_ids = self._dense_search(query_embedding, top_k=top_k)
        expanded_ids = []
        seen = set()

        for chunk_id in initial_ids:
            if chunk_id not in seen:
                expanded_ids.append(chunk_id)
                seen.add(chunk_id)
            idx = self.id_to_index.get(chunk_id)
            if idx is not None:
                current_lec = self.chunks[idx]["lecture"]
                for neighbor_idx in [idx - 1, idx + 1]:
                    if 0 <= neighbor_idx < len(self.chunks):
                        if self.chunks[neighbor_idx]["lecture"] == current_lec:
                            nid = self.chunks[neighbor_idx]["id"]
                            if nid not in seen and len(expanded_ids) < top_k * 2:
                                expanded_ids.append(nid)
                                seen.add(nid)
        return expanded_ids[:top_k]

    def retrieve_hybrid(self, query, query_embedding, top_k=5):
        dense_ids = self._dense_search(query_embedding, top_k=top_k * 2)
        sparse_ids = self._sparse_search(query, top_k=top_k * 2)
        fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)
        return fused_ids[:top_k]

    def retrieve_rerank(self, query, query_embedding, top_k=5):
        candidate_ids = self.retrieve_hybrid(query, query_embedding, top_k=top_k * 2)

        if not self.reranker or not candidate_ids:
            return candidate_ids[:top_k]

        candidate_chunks = [self.chunks[self.id_to_index[cid]] for cid in candidate_ids]
        pairs = [(query, chunk["text"]) for chunk in candidate_chunks]

        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        scored_pairs = list(zip(candidate_ids, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in scored_pairs[:top_k]]
