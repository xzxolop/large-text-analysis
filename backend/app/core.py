import nltk
import kagglehub
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import os

from .invertedindex import InvertedIndex

def create_inverted_index(_texts):
    """Создает и возвращает инвертированный индекс"""
    print("Создание инвертированного индекса...")
    index = InvertedIndex()
    index.add_documents(_texts)
    print(f"Индекс создан! Слов: {len(index.index)}, Документов: {index.doc_count}")
    return index

def load_data():
    print("Начало загрузки данных...")
    
    # Загружаем датасет через kagglehub
    path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
    path_to_comments = Path(path) / "the-reddit-dataset-dataset-comments.csv"
    
    print(f"Датасет загружен в: {path_to_comments}")

    # Читаем CSV файл
    comments_df = pd.read_csv(path_to_comments)
    comments_body = comments_df["body"]

    all_sentences = []

    # Проходим по каждому комментарию
    for comment in comments_body:
        if isinstance(comment, str):  # проверяем, что это строка
            sentences = sent_tokenize(comment)
            for sent in sentences:
                all_sentences.append(sent)

    df = pd.DataFrame()
    df['document'] = all_sentences

    # Загружаем NLTK данные
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        if not isinstance(text, str):
            return ''

        # Сначала разбиваем на предложения
        sentences = sent_tokenize(text)
        all_filtered_tokens = []
        
        for sentence in sentences:
            # Токенизируем каждое предложение отдельно
            tokens = word_tokenize(sentence.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            all_filtered_tokens.extend(filtered_tokens)
        
        # Возвращаем все токены как один текст
        return ' '.join(all_filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)
    print(f"Загрузка завершена! Получено {len(df)} предложений")
    return df