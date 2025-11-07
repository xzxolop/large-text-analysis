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
    print("start load")
    
    try:
        # Новый API kagglehub
        path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
        path_to_comments = Path(path) / "the-reddit-dataset-dataset-comments.csv"
        
        # Альтернативные варианты если выше не работает:
        # path = kagglehub.load_dataset("pavellexyr/the-reddit-dataset-dataset")
        # или используем прямое скачивание через opendatasets
        
    except Exception as e:
        print(f"Ошибка загрузки через kagglehub: {e}")
        print("Пытаемся использовать локальный файл...")
        
        # Проверяем есть ли локальный файл
        if os.path.exists("the-reddit-dataset-dataset-comments.csv"):
            path_to_comments = Path("the-reddit-dataset-dataset-comments.csv")
        else:
            # Создаем демо-данные для тестирования
            print("Создаем демо-данные...")
            return create_sample_data()

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
    print(f"Загружено {len(df)} предложений")
    print("end load")
    return df

def create_sample_data():
    """Создает демо-данные для тестирования"""
    sample_texts = [
        "This is a test sentence about programming and data science.",
        "Machine learning is amazing for data analysis.",
        "Python and JavaScript are popular programming languages.",
        "Data science involves statistics and programming.",
        "I love working with Python for data analysis.",
        "Programming requires logic and problem solving skills.",
        "Data analysis is crucial for business decisions.",
        "Machine learning algorithms can predict future trends.",
        "Python programming is versatile and powerful.",
        "JavaScript is essential for web development."
    ]
    
    df = pd.DataFrame({'document': sample_texts})
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        sentences = sent_tokenize(text)
        all_filtered_tokens = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            all_filtered_tokens.extend(filtered_tokens)
        
        return ' '.join(all_filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)
    print("Созданы демо-данные для тестирования")
    return df