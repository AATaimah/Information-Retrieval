# Assignment 1: Information Retrieval System (SciFact)

## Team
- Anas Taimah / 300228842
- Sami Diouf / 300296671
- John Ghai Manyang / 300272467

## Task Split
- Step 1 (Preprocessing): Sami
- Step 2 (Indexing): John
- Step 3 (Retrieval and Ranking): Anas
- Integration and Evaluation: Anas, Sami, John
- Report Writing: Anas, Sami, John

## Project Structure
- `src/preprocessor.py`: tokenization, filtering, stopword removal, stemming.
- `src/index.py`: inverted index with TF-IDF weights and document vector lengths.
- `src/retrieve.py`: query scoring with cosine similarity and ranked retrieval.
- `outputs/index/`: prebuilt index files.
- `Results`: ranked output file for the best run submitted.

## Dataset Note (Submission Policy)
- The initial text collection (`scifact/corpus.jsonl`) is intentionally excluded from this submission, per assignment instructions.
- This repository uses prebuilt index files in `outputs/index/`, so retrieval does not require rebuilding from the full corpus.
- To rerun retrieval/evaluation, place the following files under `scifact/`:
  - `scifact/queries.jsonl`
  - `scifact/qrels/train.tsv`
  - `scifact/qrels/test.tsv`

## Functionality, Algorithms, Data Structures, and Optimizations

### Step 1: Preprocessing
- Algorithm:
  - Lowercase tokenization (`word_tokenize`).
  - Keep alphabetic tokens only (`isalpha()`) to remove punctuation/numbers.
  - Remove English stopwords.
  - Apply Porter stemming.
- Data structures:
  - Token lists per document/query field.
  - `set` for stopwords (O(1) membership checks).
- Optimization notes:
  - Lightweight filters are applied before stemming to reduce stemmer calls.
  - Shared preprocessing function is used for both corpus and queries for consistency.

### Step 2: Indexing
- Algorithm:
  - Build inverted index over `HEAD + TEXT`.
  - For each document: compute term frequency (`tf`) with `Counter`.
  - Compute document frequency (`df`) per term.
  - Compute inverse document frequency (`idf = log(N/df)`).
  - Compute TF-IDF document vector L2 lengths (`doc_len`).
- Data structures:
  - `index`: `term -> {doc_id: tf}`
  - `df`: `term -> df`
  - `idf`: `term -> idf`
  - `doc_len`: `doc_id -> ||doc||`
  - `N`: number of documents
- Optimization notes:
  - Uses postings traversal to score only candidate documents containing query terms.
  - Prebuilt index artifacts are stored as JSON and loaded directly in retrieval.

### Step 3: Retrieval and Ranking
- Algorithm:
  - Preprocess query text.
  - Build query TF-IDF weights.
  - Compute cosine similarity:
    - numerator: dot product over shared terms only
    - denominator: `||q|| * ||d||`
  - Rank by descending score; break ties with `doc_id` for deterministic output.
- Data structures:
  - Query term counts via `Counter`.
  - Score accumulator `doc_id -> score` (`defaultdict(float)`).
- Optimization notes:
  - Candidate generation comes from postings lists, avoiding full-corpus scoring.
  - Top-k truncation to 100 documents per query.

## Vocabulary
- Vocabulary size: `18,799` terms
- Sample of 100 vocabulary tokens:
`aa, aaa, aab, aabenhu, aacr, aad, aag, aai, aam, aarhu, aaronquinlan, aasv, aatf, aauaaa, aav, ab, abad, abandon, abas, abb, abber, abbott, abbrevi, abc, abciximab, abd, abdb, abdomen, abdomin, abduct, aberr, aberrantli, abeta, abi, abil, abiot, abirateron, abl, ablat, abm, abmd, abnorm, abolish, abort, abound, abp, abpi, abrb, abroad, abrog, abrupt, abruptli, abscess, abscis, absciss, absenc, absent, absolut, absorb, absorpt, absorptiometri, abstain, abstent, abstin, abstract, abstracta, abstractmicrorna, abuja, abulia, abund, abundantli, abus, abut, abv, ac, acad, academ, academi, academia, acambi, acarbos, acasi, acc, accagccu, acceler, acceleromet, accentu, accept, acceptor, access, accessori, accid, accident, acclim, acco, accommod, accompani, accomplish, accord, accordingli`

## Complete Run Instructions

### 1) Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nltk tqdm
python3 -m nltk.downloader punkt stopwords
```

### 2) Build `trec_eval` (included in repo)
```bash
cd trec_eval
make
cd ..
```

### 3) Prepare required query/qrels files
Create this layout (without the initial text collection):
```bash
scifact/
  queries.jsonl
  qrels/
    train.tsv
    test.tsv
```

### 4) Run retrieval for required query-field settings
SciFact provides a single query text field (`text`). To satisfy the assignment requirement for query fields, `src/retrieve.py` supports:
- `title_only`: uses the first 5 raw query tokens as a title proxy.
- `title_plus_text`: concatenates title proxy + full query text.

Run A (`title_only`):
```bash
python3 src/retrieve.py --query-mode title_only --title-tokens 5 --output Results_title_only --run-tag tfidf_title_only
```

Run B (`title_plus_text`):
```bash
python3 src/retrieve.py --query-mode title_plus_text --title-tokens 5 --output Results_title_plus_text --run-tag tfidf_title_plus_text
```

### 5) Convert qrels and evaluate with `trec_eval`
```bash
awk 'NR>1 {printf "%s 0 %s %s\n", $1, $2, $3}' scifact/qrels/test.tsv > /tmp/scifact_test.qrels
./trec_eval/trec_eval -m num_q -m map /tmp/scifact_test.qrels Results_title_only
./trec_eval/trec_eval -m num_q -m map /tmp/scifact_test.qrels Results_title_plus_text
```

### 6) Keep best run in file named `Results`
```bash
cp Results_title_plus_text Results
```

## Submission Zip (without initial text collection)
Use this from repository root:
```bash
zip -r Assignment1_submission.zip README.md src outputs Results Results_title_only Results_title_plus_text trec_eval -x "*/.DS_Store" "*/__pycache__/*" "*.pyc" "trec_eval/trec_eval.dSYM/*"
```

## Query-Field Runs and Comparison
- Run A (`title_only`):
  - `num_q = 153`
  - `MAP = 0.2508`
- Run B (`title_plus_text`):
  - `num_q = 153`
  - `MAP = 0.4116`

Better run: `title_plus_text`.

## Output Format
Each line in `Results` follows:
`query_id Q0 doc_id rank score tag`

Example:
`1 Q0 13231899 1 0.097958 tfidf_title_plus_text`

## First 10 Answers (First 2 Test Queries)
Query `1`:
`1 Q0 13231899 1 0.097958 tfidf_title_plus_text`
`1 Q0 26731863 2 0.090550 tfidf_title_plus_text`
`1 Q0 994800 3 0.079766 tfidf_title_plus_text`
`1 Q0 42421723 4 0.079335 tfidf_title_plus_text`
`1 Q0 35008773 5 0.074438 tfidf_title_plus_text`
`1 Q0 12156187 6 0.066900 tfidf_title_plus_text`
`1 Q0 21439640 7 0.066169 tfidf_title_plus_text`
`1 Q0 1836154 8 0.061335 tfidf_title_plus_text`
`1 Q0 1855679 9 0.061005 tfidf_title_plus_text`
`1 Q0 10786948 10 0.056781 tfidf_title_plus_text`

Query `3`:
`3 Q0 23389795 1 0.335418 tfidf_title_plus_text`
`3 Q0 2739854 2 0.262708 tfidf_title_plus_text`
`3 Q0 4378885 3 0.230812 tfidf_title_plus_text`
`3 Q0 4632921 4 0.212386 tfidf_title_plus_text`
`3 Q0 14717500 5 0.203817 tfidf_title_plus_text`
`3 Q0 16398049 6 0.189999 tfidf_title_plus_text`
`3 Q0 18494847 7 0.184447 tfidf_title_plus_text`
`3 Q0 4414547 8 0.171669 tfidf_title_plus_text`
`3 Q0 1544804 9 0.168001 tfidf_title_plus_text`
`3 Q0 19058822 10 0.164519 tfidf_title_plus_text`

## Results Discussion
- Retrieval is run on odd-numbered query IDs, giving `547` retrieval queries.
- `trec_eval` reports `num_q = 153` because only those queries have judgments in `scifact/qrels/test.tsv`.
- MAP improves substantially from `0.2508` (title-only) to `0.4116` (title + full query text), so adding full query text gives better ranking quality in this setup.

# Assignment 2: Neural Information Retrieval System (SciFact)

## Team
- Anas Taimah / 300228842
- Sami Diouf / 300296671
- John Ghai Manyang / 300272467

## Goal
The goal of Assignment 2 is to improve the Assignment 1 IR system by applying neural language models (transformer-based) to achieve higher evaluation scores. We use the same SciFact document collection, the same test queries (odd query IDs), and the same evaluation metrics (MAP and P@10). Our approach follows a standard neural IR pipeline: use the classical IR system from Assignment 1 to retrieve up to 100 candidate documents per query, then re-rank these candidates using neural similarity models.

## Approach Overview
1. **Candidate Generation (Assignment 1 baseline)**  
   We use our TF-IDF + cosine similarity retrieval system to retrieve the top 100 candidate documents for each query using the inverted index.

2. **Neural Re-ranking (Assignment 2)**  
   We apply two neural methods to re-score and re-rank the top 100 candidates per query:
   - a **bi-encoder** sentence embedding model (fast)
   - a **cross-encoder** transformer re-ranker (slower but typically stronger)

## Neural Methods Implemented (2 Models)

### Method 1 — Bi-encoder (SentenceTransformer)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **How it works:** The query and each candidate document are encoded independently into dense embeddings. Relevance is computed via cosine similarity, then candidates are re-ranked by similarity score.
- **Output file:** `Results_minilm`

### Method 2 — Cross-encoder (Transformer Pairwise Re-ranker)
- **Model:** `cross-encoder/ms-marco-MiniLM-L6-v2`
- **How it works:** The model scores each *(query, document)* pair directly, producing a relevance score per candidate. Candidates are re-ranked by this score.
- **Output file:** `Results_cross`

## Files Produced
- `Results_minilm` — bi-encoder reranked run  
- `Results_cross` — cross-encoder reranked run  
- `Results` — best run submitted (currently the cross-encoder run)

## Evaluation Results (MAP and P@10)
We evaluated both neural methods using `trec_eval` on the SciFact test qrels.

- **Bi-encoder (all-MiniLM-L6-v2):** MAP = **0.5914**, P@10 = **0.0954**
- **Cross-encoder (ms-marco-MiniLM-L6-v2):** MAP = **0.6345**, P@10 = **0.0922** *(Best MAP)*

**Best run:** Cross-encoder (`Results_cross`), copied as `Results` for submission.

## Output Format
Each line in `Results` follows:
`query_id Q0 doc_id rank score tag`

Example:
`1 Q0 43385013 1 -6.850762 best_neural`

## First 10 Answers (Queries 1 and 3) — Best Run (`Results`)

Query `1` (top 10):
## First 10 Answers (Queries 1 and 3) — Best Run (`Results`)

Query `1` (top 10):
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

Query 3 top 10:
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

Run Instructions (Assignment 2)
1) Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install nltk tqdm numpy torch sentence-transformers scikit-learn
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

2) Required Local Dataset Files (DO NOT COMMIT)

Place the following files in the repository root (same folder as README.md):

corpus.jsonl

queries.jsonl

test.tsv

Important: Do NOT commit/push these dataset files to GitHub.

3) Run Neural Re-ranking
python src/neural_rerank.py

Expected output files (repo root):

Results_minilm

Results_cross

Results

4) Evaluate MAP and P@10 (trec_eval)
awk -F'\t' 'NR>1 {print $1 " 0 " $2 " " $3}' test.tsv > qrels_clean.txt
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_minilm
./trec_eval/trec_eval -m map -m P.10 qrels_clean.txt Results_cross

Choose the best run (typically higher MAP) and keep it as Results:
cp Results_cross Results   # or: cp Results_minilm Results   
