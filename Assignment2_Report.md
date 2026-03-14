# CSI4107
## Info Retrieval & Internet
## Assignment 2
## Neural Information Retrieval System

**Submitted by:**  
Anas Taimah # 300228842  
Sami Diouf # 300296671  
John Ghai Manyang # 300272467  

March 14, 2026  
University of Ottawa

## Introduction

In this project, we built an improved version of our Assignment 1 information retrieval system for the SciFact dataset by adding neural reranking methods. The goal of Assignment 2 was to move beyond classical TF-IDF retrieval and test whether transformer-based neural models could improve ranking quality on the same collection, queries, and evaluation setup. Our final system follows a two-stage retrieval pipeline. First, we use the Assignment 1 TF-IDF inverted-index system to retrieve up to 100 candidate documents for each query. Then, we re-rank those candidates using two neural methods: a bi-encoder model and a cross-encoder model. This approach lets us keep the efficiency of classical candidate generation while benefiting from stronger neural relevance modeling.

Link to GitHub repository: <https://github.com/AATaimah/Information-Retrieval>

## Dataset

We used the same SciFact dataset as in Assignment 1. The SciFact benchmark contains a scientific document collection, natural-language claims used as queries, and relevance judgments for evaluation.

For this assignment, the local files used by our code are:
- `corpus.jsonl` in the repository root
- `queries.jsonl` in the repository root
- `test.tsv` in the repository root

The collection contains **5,183 documents**. Queries are read from `queries.jsonl`, and, following the assignment instructions, we use only the **odd-numbered query IDs** (`1, 3, 5, ...`) for retrieval. This gives **547 retrieval queries** in total. Evaluation is performed with the SciFact test relevance judgments from `test.tsv`.

The candidate-generation stage reuses the Assignment 1 inverted index built over the SciFact corpus. That baseline index contains a vocabulary of **18,799 unique terms** after preprocessing.

## Methods and Setup

### Baseline Candidate Generation

As the first stage of the system, we reuse our Assignment 1 retrieval pipeline. Queries are preprocessed with tokenization, stopword removal, filtering of punctuation and numbers, and Porter stemming. Using the prebuilt inverted index, we compute TF-IDF query weights and cosine similarity scores against documents in the collection. Instead of ranking the entire collection directly for final output, this baseline stage is used to retrieve the **top 100 candidate documents per query**, which are then passed to the neural reranking models.

This two-stage setup is important because scoring every query against every document with transformer models would be much slower. By restricting neural scoring to a small candidate set, we significantly reduce computation while still allowing the neural models to improve ranking quality.

### Method 1 - Bi-encoder

Our first neural retrieval method uses the SentenceTransformer bi-encoder model `sentence-transformers/all-MiniLM-L6-v2`. In this approach, the query and each candidate document are encoded independently into dense vector embeddings. After normalizing the embeddings, we compute similarity scores using the dot product, which is equivalent to cosine similarity on normalized vectors. The candidate documents are then re-ranked in descending order of similarity.

This method is relatively efficient because document representations are produced independently rather than jointly with the query. In our implementation, the model is loaded once, the query is encoded once per query, and candidate documents are encoded in batches of 16.

### Method 2 - Cross-encoder

Our second neural retrieval method uses the cross-encoder model `cross-encoder/ms-marco-MiniLM-L6-v2`. Unlike the bi-encoder, the cross-encoder directly scores each `(query, document)` pair with a transformer. This allows the model to capture richer interactions between query terms and document text, usually resulting in better ranking quality.

For each query, we create query-document pairs for the top 100 TF-IDF candidates and pass them to the cross-encoder. The model returns a relevance score for each pair, and the documents are then re-ranked by these scores. This method is slower than the bi-encoder because every query-document pair must be processed jointly, but it is typically more accurate.

### Document Representation

For neural reranking, each SciFact document is converted to a text string by concatenating the document title and abstract text. To reduce runtime and memory cost, document text is truncated to the first **1500 characters** before reranking.

### Data Structures

The implementation uses the following main data structures:
- an inverted index from Assignment 1: `term -> {doc_id: tf}`
- dictionaries for document frequency, inverse document frequency, and document vector lengths
- a dictionary `doc_id -> {"title", "text"}` for loading the SciFact corpus in memory
- `Counter` objects for query term frequencies
- `defaultdict(float)` for TF-IDF score accumulation
- ranking lists of `(doc_id, score)` tuples for both neural methods

### Optimizations

We included several practical optimizations to improve efficiency and overall retrieval quality:
- reuse of the Assignment 1 TF-IDF index for fast candidate generation
- restriction of neural scoring to the top 100 baseline candidates per query
- loading each transformer model only once instead of once per query
- batching neural inference with `batch_size=16`
- truncating document text to 1500 characters before neural scoring
- deterministic sorting by score and document ID to produce stable output files

## Integration and Evaluation

### Setup

We evaluated both neural reranking methods on the SciFact dataset using the same testing procedure as in Assignment 1. Queries were read from `queries.jsonl`, filtered to odd-numbered IDs, and reranked after initial TF-IDF candidate generation. For every query, the system outputs up to the top 100 ranked documents in standard TREC format:

`query_id Q0 doc_id rank score tag`

The output files produced by the system are:
- `Results_minilm` for the bi-encoder run
- `Results_cross` for the cross-encoder run
- `Results` for the best run submitted

### Evaluation Protocol

We evaluated both methods with `trec_eval` using the SciFact test qrels. Since `test.tsv` is tab-separated and contains a header row, we first converted it into TREC qrels format:

```bash
awk -F'\t' 'NR>1 {print $1 " 0 " $2 " " $3}' test.tsv > qrels_clean.txt
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_minilm
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_cross
```

As in Assignment 1, `trec_eval` only evaluates queries that have relevance judgments. Therefore, although we retrieve results for 547 odd-numbered queries, the reported scores are based on **153 judged test queries**.

## Results

We report both **MAP (Mean Average Precision)** and **P@10**, as required in the assignment.

- `num_q = 153`
- **Bi-encoder (`all-MiniLM-L6-v2`)**: `MAP = 0.5914`, `P@10 = 0.0954`
- **Cross-encoder (`ms-marco-MiniLM-L6-v2`)**: `MAP = 0.6345`, `P@10 = 0.0922`

The best run is the **cross-encoder**, which we copied to `Results` for submission.

## Assignment 1 vs Assignment 2 Comparison

Our best Assignment 1 system was the TF-IDF `title_plus_text` configuration, which achieved:
- **Assignment 1 best MAP = 0.4116**

For Assignment 2:
- **Bi-encoder MAP = 0.5914**
- **Cross-encoder MAP = 0.6345**

This shows a substantial improvement from Assignment 1 to Assignment 2:

`0.4116 -> 0.6345`

The cross-encoder achieved the best MAP because it scores each query-document pair jointly, allowing it to capture stronger relevance signals than classical TF-IDF retrieval or embedding-only reranking. The main tradeoff is speed. The cross-encoder is more computationally expensive than the bi-encoder because it must process every query-document pair together. In contrast, the bi-encoder is faster and still improves significantly over the Assignment 1 baseline, but it does not reach the same accuracy as the cross-encoder.

## First 10 Answers to Queries 1 and 3

Below are the first 10 answers for queries 1 and 3 from the best run (`Results`).

Query `1`:

```text
1 Q0 43385013 1 -6.850762 best_neural
1 Q0 37437064 2 -7.243060 best_neural
1 Q0 1065627 3 -8.584124 best_neural
1 Q0 10906636 4 -8.703024 best_neural
1 Q0 17518195 5 -8.954710 best_neural
1 Q0 36637129 6 -9.017780 best_neural
1 Q0 39851630 7 -9.032077 best_neural
1 Q0 40769868 8 -9.187998 best_neural
1 Q0 10931595 9 -9.357861 best_neural
1 Q0 19651306 10 -9.417481 best_neural
```

Query `3`:

```text
3 Q0 14717500 1 2.242803 best_neural
3 Q0 4414547 2 1.503803 best_neural
3 Q0 4632921 3 0.383114 best_neural
3 Q0 2739854 4 0.122495 best_neural
3 Q0 19058822 5 -0.242292 best_neural
3 Q0 23389795 6 -0.622116 best_neural
3 Q0 4378885 7 -1.279230 best_neural
3 Q0 1388704 8 -1.543448 best_neural
3 Q0 13519661 9 -1.859299 best_neural
3 Q0 2107238 10 -2.462658 best_neural
```

## Task Distribution

- Sami: data preparation, environment setup notes, and report writing
- John: baseline index and candidate-generation integration, and report writing
- Anas: neural reranking implementation, evaluation of results, and report writing

## Complete Instructions to Run the Programs

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install nltk tqdm numpy torch sentence-transformers scikit-learn
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 2. Required Local Files

Place the following files in the repository root:

```text
corpus.jsonl
queries.jsonl
test.tsv
```

### 3. Run the Neural Reranking Pipeline

```bash
python src/neural_rerank.py
```

This produces:

```text
Results_minilm
Results_cross
Results
```

### 4. Evaluate the Runs

```bash
awk -F'\t' 'NR>1 {print $1 " 0 " $2 " " $3}' test.tsv > qrels_clean.txt
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_minilm
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_cross
```

## Conclusion

In this assignment, we successfully extended our Assignment 1 retrieval system with neural reranking methods and achieved substantially better evaluation scores. Using the Assignment 1 TF-IDF system as a candidate generator, we tested two neural approaches: a bi-encoder and a cross-encoder. Both methods outperformed our Assignment 1 baseline, and the cross-encoder achieved the best overall MAP score of **0.6345**.

This project showed that neural information retrieval methods can significantly improve ranking quality over classical vector-space retrieval, especially when used in a two-stage pipeline. At the same time, the results also show the tradeoff between efficiency and effectiveness: the cross-encoder gave the highest accuracy, while the bi-encoder provided a faster but slightly weaker alternative. Overall, this work gave us practical experience with modern neural IR systems and how they can be combined with classical indexing and retrieval methods.
