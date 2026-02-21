from sklearn.feature_extraction.text import TfidfVectorizer

# Пример текстовых данных
documents = [
    "Машинное обучение - это интересная область.",
    "Обучение с учителем - ключевой аспект машинного обучения.",
    "Область NLP также связана с машинным обучением."
]

# Создание объекта TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Применение TF-IDF к текстовым данным
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Получение списка ключевых слов и их значения TF-IDF для первого документа
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]

# Сортировка слов по значениям TF-IDF
sorted_keywords = [word for _, word in sorted(zip(tfidf_scores, feature_names), reverse=True)]

print("Ключевые слова:", sorted_keywords)


from sklearn.cluster import KMeans
import numpy as np

# Применение кластеризации KMeans к матрице TF-IDF
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(tfidf_matrix)

# Показать примеры документов в каждом кластере
for cluster_id in range(num_clusters):
    cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
    print(f"Кластер {cluster_id + 1}:")
    for idx in cluster_indices:
        print(documents[idx])
    print("--------")