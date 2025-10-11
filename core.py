import nltk
import utils
import math

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

# TODO: при токенизации необходимо не добавлять запятые, точки, и т.д.
def calculate_TF(word, document):
    words = nltk.word_tokenize(document)
    include_count = utils.count_word_matches(word, words)

    #print(include_count)
    #print(words)
    #print(len(words))

    return include_count / len(words)

def claculate_IDF(word, documents):
    return math.log(len(documents) / count_documents_include_word(word, documents))

def count_documents_include_word(word, documents):
    cnt = 0
    for d in documents:
        if d.count(word):
            cnt+=1
    return cnt

def make_map_most_popular(search_word: str, documents):
    m = {}
    for doc in documents[:100]:
        words = nltk.word_tokenize(doc)
        if search_word in words:
            for word in words:
                if word != search_word:
                    if word in m:
                        m[word] = (m[word] + 1)
                    else:
                        m[word] = 1
    return m
