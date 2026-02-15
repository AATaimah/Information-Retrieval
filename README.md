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
- `scifact/`: dataset files (`corpus.jsonl`, `queries.jsonl`, `qrels/`).
- `outputs/index/`: prebuilt index files.
- `Results`: ranked output file for test (odd-numbered) queries.

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nltk tqdm
python3 -m nltk.downloader punkt stopwords
```

## Run Retrieval
This run uses:
- queries from `scifact/queries.jsonl`
- prebuilt index from `outputs/index/scifact_head_text_*`
- output file `Results`

```bash
python3 src/retrieve.py
```

## Evaluate with trec_eval (optional)
Build and run `trec_eval` from the included source folder:
```bash
cd trec_eval
make
cd ..
awk 'NR>1 {printf "%s 0 %s %s\n", $1, $2, $3}' scifact/qrels/test.tsv > /tmp/scifact_test.qrels
./trec_eval/trec_eval -m num_q -m map /tmp/scifact_test.qrels Results
```

## Output Format
Each line in `Results` follows:
`query_id Q0 doc_id rank score tag`

Example:
`1 Q0 13231899 1 0.102204 tfidf_cosine`

## Notes
- Only odd-numbered query IDs are used for testing (1, 3, 5, ...), sorted ascending.
- `Results` includes top-100 retrieved documents per test query.

## Functionality Details
- Preprocessing: lowercase tokenization, punctuation/number filtering (`isalpha()`), English stopword removal, Porter stemming.
- Indexing: inverted index over `HEAD` + `TEXT`, storing postings as `term -> {doc_id: tf}`, plus `df`, `idf = log(N/df)`, and per-document TF-IDF vector lengths.
- Retrieval: query preprocessing + TF-IDF query vector; cosine similarity scoring against indexed documents; descending score ranking; deterministic tie-break by `doc_id`.
- Test query selection: only odd query IDs from `scifact/queries.jsonl`.

## Vocabulary
- Vocabulary size: `18,799` terms
- Sample of 100 vocabulary tokens:
`aa, aaa, aab, aabenhu, aacr, aad, aag, aai, aam, aarhu, aaronquinlan, aasv, aatf, aauaaa, aav, ab, abad, abandon, abas, abb, abber, abbott, abbrevi, abc, abciximab, abd, abdb, abdomen, abdomin, abduct, aberr, aberrantli, abeta, abi, abil, abiot, abirateron, abl, ablat, abm, abmd, abnorm, abolish, abort, abound, abp, abpi, abrb, abroad, abrog, abrupt, abruptli, abscess, abscis, absciss, absenc, absent, absolut, absorb, absorpt, absorptiometri, abstain, abstent, abstin, abstract, abstracta, abstractmicrorna, abuja, abulia, abund, abundantli, abus, abut, abv, ac, acad, academ, academi, academia, acambi, acarbos, acasi, acc, accagccu, acceler, acceleromet, accentu, accept, acceptor, access, accessori, accid, accident, acclim, acco, accommod, accompani, accomplish, accord, accordingli`

## First 10 Answers (First 2 Test Queries)
Query `1`:
`1 Q0 13231899 1 0.102204 tfidf_cosine`
`1 Q0 10906636 2 0.097901 tfidf_cosine`
`1 Q0 26731863 3 0.094475 tfidf_cosine`
`1 Q0 994800 4 0.083224 tfidf_cosine`
`1 Q0 42421723 5 0.082774 tfidf_cosine`
`1 Q0 35008773 6 0.077665 tfidf_cosine`
`1 Q0 12156187 7 0.069800 tfidf_cosine`
`1 Q0 21439640 8 0.069037 tfidf_cosine`
`1 Q0 26071782 9 0.066504 tfidf_cosine`
`1 Q0 1855679 10 0.063650 tfidf_cosine`

Query `3`:
`3 Q0 23389795 1 0.418710 tfidf_cosine`
`3 Q0 2739854 2 0.327945 tfidf_cosine`
`3 Q0 4632921 3 0.265127 tfidf_cosine`
`3 Q0 14717500 4 0.254430 tfidf_cosine`
`3 Q0 4378885 5 0.225390 tfidf_cosine`
`3 Q0 4414547 6 0.214299 tfidf_cosine`
`3 Q0 15153602 7 0.186112 tfidf_cosine`
`3 Q0 10279084 8 0.179770 tfidf_cosine`
`3 Q0 4427060 9 0.176127 tfidf_cosine`
`3 Q0 19058822 10 0.172698 tfidf_cosine`

## Evaluation
- MAP on judged test queries: `0.4246`
- Number of judged test queries evaluated (`num_q`): `153`
- `trec_eval` command:
```bash
awk 'NR>1 {printf "%s 0 %s %s\n", $1, $2, $3}' scifact/qrels/test.tsv > /tmp/scifact_test.qrels
./trec_eval/trec_eval -m num_q -m map /tmp/scifact_test.qrels Results
```

## Query-Field Runs Note
- Current SciFact query file used in this project (`scifact/queries.jsonl`) has a single main text field (`text`) per query claim.
- This repository currently includes one submitted run (`tfidf_cosine`) using that query text.
