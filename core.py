import nltk

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


    