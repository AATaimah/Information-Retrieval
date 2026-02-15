import json
import math
from collections import defaultdict, Counter
from pathlib import Path

from index import InvertedIndex
from preprocessor import preprocess_text


def load_queries(path: Path) -> list[tuple[str, str]]:
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


def score_query(index: InvertedIndex, q_tokens: list[str]) -> dict[str, float]:
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


def main():
    root = Path(__file__).resolve().parent.parent
    index_prefix = root / "outputs" / "index" / "scifact_head_text"
    queries_path = root / "scifact" / "queries.jsonl"
    output_path = root / "Results"
    run_tag = "tfidf_cosine"

    index = InvertedIndex.load(str(index_prefix))
    queries = load_queries(queries_path)
    # Keep only odd-numbered queries and sort ascending by numeric id.
    test_queries = [q for q in queries if is_odd_query_id(q[0])]
    test_queries.sort(key=lambda x: int(x[0]))

    with output_path.open("w", encoding="utf-8") as out:
        for qid, text in test_queries:
            q_tokens = preprocess_text(text)
            scores = score_query(index, q_tokens)
            # sort by score desc, then doc_id for determinism
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


if __name__ == "__main__":
    main()
