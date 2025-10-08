import kagglehub
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import core
import utils

# Note: Необходимо выполнить при первом запуске
path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
nltk.download('punkt_tab')

path = "C:/Users/evoro/.cache/kagglehub/datasets/pavellexyr/the-reddit-dataset-dataset/versions/1"
print("Path to dataset files:", path)

posts_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-posts.csv")
comments_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-comments.csv")
comments_body = comments_df["body"]
comments_body = comments_body.fillna('')
comments_arr = comments_body.tolist()
print(comments_arr[2])

sentences = core.tokenize_sentances(comments_arr)
words = core.tokenize_words(sentences)

tf = core.calculate_TF("I", comments_arr[2])
print("tf:", tf )

idf = core.claculate_IDF("I", comments_arr)
print("idf:", idf)

print("tf-idf:", tf*idf)

search_word = "data"
cnt_words = utils.count_word_matches(search_word, words)
print(f"Колличество слов {search_word}: {cnt_words}")



# Создаем TF-IDF векторизатор с настройками
vectorizer = TfidfVectorizer(
    lowercase=False,      # Отключаем приведение к нижнему регистру
    stop_words=None,      # Отключаем удаление стоп-слов
    token_pattern=r'(?u)\b\w+\b'  # Шаблон для токенизации
)

# Обучаем модель и преобразуем предложения
tfidf_matrix = vectorizer.fit_transform(comments_arr)

# Получаем список всех слов (features)
feature_names = vectorizer.get_feature_names_out()

# Проверяем, есть ли слово в списке
print(f"Всего уникальных слов: {len(feature_names)}")
print("Первые 20 слов:", feature_names[:20])

# Обучаем на всех документах чтобы получить словарь
vectorizer.fit(comments_arr)

# Преобразуем только третье предложение
tf_matrix = vectorizer.transform([comments_arr[2]])

# Получаем названия фич (слов)
feature_names = vectorizer.get_feature_names_out()

# Создаем DataFrame для удобного просмотра
tf_df = pd.DataFrame(
    tf_matrix.toarray(),
    columns=feature_names,
    index=['Sentence 3']
)

# Находим TF для конкретного слова (например, "I")
if 'I' in feature_names:
    word_index = list(feature_names).index('I')
    tf_value = tf_matrix[0, word_index]
    print(f"TF для слова 'I' в третьем предложении: {tf_value}")
else:
    print("Слово 'I' не найдено")

# Выводим все ненулевые TF значения для третьего предложения
print("\nВсе ненулевые TF значения для третьего предложения:")
non_zero_tf = tf_df.loc['Sentence 3'][tf_df.loc['Sentence 3'] > 0]
print(non_zero_tf.sort_values(ascending=False))

