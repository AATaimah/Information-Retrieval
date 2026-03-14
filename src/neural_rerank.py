from tqdm import tqdm
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

from index import InvertedIndex
from preprocessor import preprocess_text

# Neural models
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_scifact_queries(path: Path) -> list[tuple[str, str]]:
    """SciFact queries.jsonl: each line has _id and text"""
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id"))
            text = obj.get("text", "")
            queries.append((qid, text))
    return queries


def is_odd_query_id(qid: str) -> bool:
    try:
        return int(qid) % 2 == 1
    except ValueError:
        return False

def score_query_tfidf_cosine(index: InvertedIndex, q_tokens: list[str]) -> dict[str, float]:
    tf_q = Counter(q_tokens)
    scores = defaultdict(float)
    q_len_sq = 0.0

    for term, tf in tf_q.items():
        if term not in index.idf:
            continue
        idf = index.idf[term]
        w_q = tf * idf
        q_len_sq += w_q * w_q
        postings = index.get_postings(term)
        for doc_id, tf_d in postings.items():
            scores[doc_id] += w_q * (tf_d * idf)

    q_len = math.sqrt(q_len_sq)
    if q_len == 0:
        return {}

    for doc_id, score in list(scores.items()):
        d_len = index.doc_len.get(doc_id, 0.0)
        if d_len == 0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] = score / (q_len * d_len)

    return scores


def get_candidates(index: InvertedIndex, q_text: str, k: int = 100) -> list[tuple[str, float]]:
    """Get top-k candidate docs using your A1 retrieval."""
    q_tokens = preprocess_text(q_text)
    scores = score_query_tfidf_cosine(index, q_tokens)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    return ranked


#  Utilities to load  docs for the neural reranking
def load_scifact_corpus(path: Path) -> dict[str, dict]:
    """
    SciFact corpus.jsonl: each line has _id, title, text, metadata.
    Return dict doc_id -> {"title":..., "text":...}
    """
    docs = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id"))
            docs[doc_id] = {
                "title": obj.get("title", ""),
                "text": obj.get("text", "")
            }
    return docs


def doc_to_string(doc: dict) -> str:
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    # Truncate long abstracts to keep reranking latency manageable.
    text = text[:1500]
    if title and text:
        return title + ". " + text
    return title or text


def write_trec_results(out_path: Path, run_tag: str, all_rankings: dict[str, list[tuple[str, float]]]):
    """
    all_rankings: qid -> list of (doc_id, score) already sorted desc
    """
    with out_path.open("w", encoding="utf-8") as out:
        for qid in sorted(all_rankings.keys(), key=lambda x: int(x)):
            ranked = all_rankings[qid][:100]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


def main():
    root = Path(__file__).resolve().parent.parent

    index_prefix = root / "outputs" / "index" / "scifact_head_text"
    queries_path = root / "queries.jsonl"
    corpus_path = root / "corpus.jsonl"

    out_biencoder = root / "Results_minilm"
    out_cross = root / "Results_cross"
    out_best = root / "Results"

    biencoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    crossencoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"

    print("Loading index...")
    index = InvertedIndex.load(str(index_prefix))

    print("Loading queries...")
    queries = load_scifact_queries(queries_path)
    test_queries = [q for q in queries if is_odd_query_id(q[0])]
    test_queries.sort(key=lambda x: int(x[0]))

    print("Loading corpus...")
    docs = load_scifact_corpus(corpus_path)

    # Bi-encoder rerank
    print("Running bi-encoder rerank...")
    biencoder_rankings = {}

    bi_model = SentenceTransformer(biencoder_model)

    for qid, qtext in tqdm(test_queries, desc="Bi-encoder queries"):
        candidates = get_candidates(index, qtext, k=100)
        cand_ids = [doc_id for doc_id, _ in candidates]
        cand_texts = [doc_to_string(docs.get(doc_id, {})) for doc_id in cand_ids]

        q_emb = bi_model.encode([qtext], normalize_embeddings=True)[0]
        d_emb = bi_model.encode(cand_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        scores = (d_emb @ q_emb).tolist()

        reranked = sorted(zip(cand_ids, scores), key=lambda x: (-x[1], x[0]))
        biencoder_rankings[qid] = reranked

    write_trec_results(out_biencoder, "biencoder_minilm", biencoder_rankings)
    print(f"Wrote {out_biencoder}")

    # Cross-encoder rerank
    print("Running cross-encoder rerank (slower)...")
    cross_rankings = {}

    ce_model = CrossEncoder(crossencoder_model)

    for qid, qtext in tqdm(test_queries, desc="Cross-encoder queries"):
        candidates = get_candidates(index, qtext, k=100)
        cand_ids = [doc_id for doc_id, _ in candidates]
        cand_texts = [doc_to_string(docs.get(doc_id, {})) for doc_id in cand_ids]

        pairs = [[qtext, dt] for dt in cand_texts]
        scores = ce_model.predict(pairs, batch_size=16).tolist()

        reranked = sorted(zip(cand_ids, scores), key=lambda x: (-x[1], x[0]))
        cross_rankings[qid] = reranked

    write_trec_results(out_cross, "crossencoder_msmarco", cross_rankings)
    print(f"Wrote {out_cross}")

    # Keep the current best-performing run in the required submission filename.
    write_trec_results(out_best, "best_neural", cross_rankings)
    print(f"Wrote {out_best} (currently same as Results_cross)")


if __name__ == "__main__":
    main()
