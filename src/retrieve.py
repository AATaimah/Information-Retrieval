import json
import math
import argparse
import re
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


def build_title_proxy(text: str, title_tokens: int = 5) -> str:
    """
    Build a lightweight 'title' proxy from the first N alphanumeric tokens.
    SciFact queries only expose a single text field, so this allows a
    title-only vs title+text comparison run for assignment requirements.
    """
    words = re.findall(r"[A-Za-z0-9]+", text)
    return " ".join(words[:title_tokens])


def query_tokens_from_mode(raw_text: str, mode: str, title_tokens: int) -> list[str]:
    title_text = build_title_proxy(raw_text, title_tokens=title_tokens)

    if mode == "title_only":
        return preprocess_text(title_text)
    if mode == "title_plus_text":
        return preprocess_text(f"{title_text} {raw_text}".strip())
    return preprocess_text(raw_text)


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
    parser = argparse.ArgumentParser(description="SciFact TF-IDF retrieval")
    parser.add_argument(
        "--query-mode",
        choices=["text_only", "title_only", "title_plus_text"],
        default="text_only",
        help="How query text is formed before preprocessing",
    )
    parser.add_argument(
        "--title-tokens",
        type=int,
        default=5,
        help="Number of leading raw tokens used to build title proxy",
    )
    parser.add_argument(
        "--output",
        default="Results",
        help="Output run file path (default: Results at repo root)",
    )
    parser.add_argument(
        "--run-tag",
        default="tfidf_cosine",
        help="Run tag written in TREC output",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    index_prefix = root / "outputs" / "index" / "scifact_head_text"
    queries_path = root / "scifact" / "queries.jsonl"
    output_path = root / args.output
    run_tag = args.run_tag

    index = InvertedIndex.load(str(index_prefix))
    queries = load_queries(queries_path)
    # Keep only odd-numbered queries and sort ascending by numeric id.
    test_queries = [q for q in queries if is_odd_query_id(q[0])]
    test_queries.sort(key=lambda x: int(x[0]))

    with output_path.open("w", encoding="utf-8") as out:
        for qid, text in test_queries:
            q_tokens = query_tokens_from_mode(
                raw_text=text,
                mode=args.query_mode,
                title_tokens=args.title_tokens,
            )
            scores = score_query(index, q_tokens)
            # sort by score desc, then doc_id for determinism
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


if __name__ == "__main__":
    main()
