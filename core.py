import nltk
import utils

def tokenize_sentances(text):
    sentences = []
    for x in text:
        if isinstance(x, str):
            sentences += nltk.sent_tokenize(x)
    return sentences

def tokenize_words(sentences):
    words = []
    for x in sentences:
        words += nltk.word_tokenize(x)
    return words

def calculate_TF(word, document):
    words = nltk.word_tokenize(document)
    include_count = utils.count_word_matches(word, words)

    print(include_count)
    print(words)
    print(len(words))

    return include_count / len(words)

def count_documents_include_word(word, documents):
    cnt = 0
    for d in documents:
        if d.count(word):
            cnt+=1
    return cnt
