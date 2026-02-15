import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
try:
    from utils import progress_bar
except Exception:
    def progress_bar(*_args, **_kwargs):
        return None
import time
import json


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize(text):
    # tokenize and lowercase the text
    return word_tokenize(text.lower())

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def remove_extras(tokens):
    # remove weird tokens that come from query fields
    return [t for t in tokens if t not in ['no_queri', 'no_narr']]

def is_valid_token(token):
    # keep only alphabetic tokens (no numbers or punctuation)
    return token.isalpha()

def preprocess_text(text):
    tokens = tokenize(text)

    # this will remove punctuation and numbers
    tokens = [t for t in tokens if is_valid_token(t)]

    # and this will remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # apply stemming
    tokens = stem_tokens(tokens)

    # this will remove stopwords again just in case
    tokens = [t for t in tokens if t not in stop_words]

    tokens = remove_extras(tokens)
    return tokens

def preprocess_documents(documents):
    previousId = "t"
    count = 1
    start_time = time.time()

    for doc in documents:
        fileId = str(doc['DOCNO'].split(" ")[0])

        if fileId != previousId:
            progress_bar(count, len(documents))
            previousId = fileId
            count += 1

        doc['TEXT'] = preprocess_text(doc['TEXT'])
        doc['HEAD'] = preprocess_text(doc['HEAD'])

    end_time = time.time()
    print(f"\nTime taken to parse and preprocess documents: {end_time - start_time:.2f} seconds")
    return documents

def preprocess_queries(queries):
    for query in queries:
        query['title'] = preprocess_text(query['title'])
        query['query'] = preprocess_text(query['query'])
        query['narrative'] = preprocess_text(query['narrative'])
    return queries

def save_preprocessed_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_preprocessed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
