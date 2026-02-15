import math
import json
from collections import defaultdict, Counter
from typing import Dict, Tuple, Iterable, Any


class InvertedIndex:
    """
    Inverted index for vector space retrieval (TF-IDF + cosine).

    Stores:
      - index: term -> {doc_id: tf}
      - df:    term -> document frequency
      - idf:   term -> inverse document frequency
      - doc_len: doc_id -> TF-IDF vector length (L2 norm)
      - N: number of documents
    """

    def __init__(self):
        self.index = defaultdict(dict)   # term -> {doc_id: tf}
        self.df = defaultdict(int)       # term -> df
        self.idf = {}                    # term -> idf
        self.doc_len = {}                # doc_id -> ||doc|| (TF-IDF)
        self.N = 0

    def build(self, documents: Iterable[Dict[str, Any]],
              use_fields=("HEAD", "TEXT"),
              doc_id_field="DOCNO",
              smooth_idf: bool = False):
        """
        Build index from preprocessed documents.

        documents: iterable of dicts. Each doc must have DOCNO and token lists in chosen fields.
        use_fields: which fields to index, e.g. ("TEXT",) or ("HEAD","TEXT")
        smooth_idf: if True, use log((N+1)/(df+1)) + 1
        """
        documents = list(documents)
        self.N = len(documents)

        # PASS 1: postings + df
        for doc in documents:
            doc_id = str(doc[doc_id_field])

            tokens = []
            for field in use_fields:
                if field in doc:
                    tokens.extend(doc[field])  # field must already be a list of tokens

            tf_counter = Counter(tokens)

            for term, tf in tf_counter.items():
                self.index[term][doc_id] = tf

            for term in tf_counter.keys():
                self.df[term] += 1

        # IDF
        self.idf = {}
        if smooth_idf:
            for term, df_val in self.df.items():
                self.idf[term] = math.log((self.N + 1) / (df_val + 1)) + 1.0
        else:
            for term, df_val in self.df.items():
                self.idf[term] = math.log(self.N / df_val)

        # PASS 2: doc lengths
        doc_len_sq = defaultdict(float)
        for term, postings in self.index.items():
            idf_val = self.idf[term]
            for doc_id, tf in postings.items():
                w = tf * idf_val
                doc_len_sq[doc_id] += w * w

        self.doc_len = {doc_id: math.sqrt(v) for doc_id, v in doc_len_sq.items()}
        return self

    def get_postings(self, term: str) -> Dict[str, int]:
        """Return postings dict doc_id -> tf (empty dict if term not found)."""
        return self.index.get(term, {})

    def save(self, prefix_path: str):
        """Save index structures to JSON files with the given prefix."""
        with open(prefix_path + "_index.json", "w", encoding="utf-8") as f:
            json.dump({t: d for t, d in self.index.items()}, f, ensure_ascii=False)

        with open(prefix_path + "_df.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.df), f, ensure_ascii=False)

        with open(prefix_path + "_idf.json", "w", encoding="utf-8") as f:
            json.dump(self.idf, f, ensure_ascii=False)

        with open(prefix_path + "_doclen.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_len, f, ensure_ascii=False)

        with open(prefix_path + "_meta.json", "w", encoding="utf-8") as f:
            json.dump({"N": self.N}, f, ensure_ascii=False)

    @staticmethod
    def load(prefix_path: str) -> "InvertedIndex":
        """Load index structures from JSON files saved with the given prefix."""
        idx = InvertedIndex()

        with open(prefix_path + "_index.json", "r", encoding="utf-8") as f:
            idx.index = defaultdict(dict, json.load(f))

        with open(prefix_path + "_df.json", "r", encoding="utf-8") as f:
            idx.df = defaultdict(int, json.load(f))

        with open(prefix_path + "_idf.json", "r", encoding="utf-8") as f:
            idx.idf = json.load(f)

        with open(prefix_path + "_doclen.json", "r", encoding="utf-8") as f:
            idx.doc_len = json.load(f)

        with open(prefix_path + "_meta.json", "r", encoding="utf-8") as f:
            idx.N = json.load(f)["N"]

        return idx
