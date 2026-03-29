from sklearn.feature_extraction.text import TfidfVectorizer
from interface.invindex import InvIndex
from nltk import word_tokenize
from collections import Counter
import math

class  ExclusiveClustererV2:

    def __init__(self, sent: list,):
        self._sent = sent
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(self._sent)
        self._feature_names = self._vectorizer.get_feature_names_out()
        self.word_doc_freq: Counter = Counter()

        for sent in self._sent:
            words = word_tokenize(sent.lower())
            
            # Учитываем только уникальные слова в предложении для doc_freq
            unique_words = set(words)
            for w in unique_words:
                self.word_doc_freq[w] += 1

    def get_clusters(self):
        my_dict = dict()

        for i in range(len(self._sent)):
            row = self._matrix[i].toarray().flatten()
            result = [(word, score) for word, score in zip(self._feature_names, row) if score > 0]
            result_sorted = sorted(result, key=lambda x: x[1], reverse=True)

            if len(result_sorted) < 2:
                continue

            word = result_sorted[0][0]
            tf_idf = result_sorted[0][1]
            freq = self.word_doc_freq.get(word, 0)
            score = tf_idf * math.log2(freq)


            if word in my_dict:
                    my_dict[word].add(score)
            else:
                my_dict[word] = {score}

        total = sum(len(s) for s in my_dict.values())
        print(total)
        index = InvIndex(my_dict)
        return index
