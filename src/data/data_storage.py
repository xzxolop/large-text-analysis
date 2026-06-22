import nltk
import kagglehub
from pathlib import Path
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from config import (
    ACTIVE_DATASET,
    DATASET_MAX_DOCUMENTS,
    FILES_DIR,
    INCLUDE_DEMO_SENTENCES,
    PROJECT_ROOT,
)
from data.demo_sentences import DEMO_SENTENCES, DEMO_SENTENCES_VERSION
from data.dataset_sources import DatasetSource, get_dataset_source
from data.sqlite_repository import SQLiteSentenceRepository


PREPROCESSING_VERSION = "1"


class DataStorage:
    """
    Хранит данные датасета в списках python.
    __main_text_list        - список документов (текстов) датасета.\n
    __orig_sent_list        - список оригинальных предложений, получается из разбиения документов датасета на предложения.\n
    __processed_sent_list   - список обработаных предложений (равно числу оригинальных.\n
    __alias_list            - список связей, где index соответствует оригинальному и обработаному предожению, а value - документу датасета.\n
    """
    __main_text_list: list[str]  #TODO: rename to documents_list
    __orig_sent_list: list[str]
    __processed_sent_list: list[str]
    __alias_list: list[int]
    __stop_words: set[str]

    def __init__(self, database_path: Path | None = None) -> None:
        self.__main_text_list = []
        self.__orig_sent_list = []
        self.__processed_sent_list = []
        self.__alias_list = []
        self.__stop_words = set()
        self.__dataset_source = get_dataset_source(ACTIVE_DATASET)

        files_path = Path(FILES_DIR)
        if not files_path.is_absolute():
            files_path = PROJECT_ROOT / files_path
        self.__repository = SQLiteSentenceRepository(
            database_path or files_path / "large_text_analysis.sqlite3"
        )

    def load_data(
        self,
        dataset_path: Path | None = None,
        dataset_name: str | None = None,
        dataset_key: str | None = None,
        max_documents: int | None = None,
        include_demo_sentences: bool | None = None,
    ):
        """Загружает данные из датасета в списки python."""

        self.__dataset_source = get_dataset_source(dataset_key or ACTIVE_DATASET)
        max_documents = self.__resolve_max_documents(max_documents)
        if include_demo_sentences is None:
            include_demo_sentences = INCLUDE_DEMO_SENTENCES

        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        else:
            # Сначала ищем кэшированный датасет
            dataset_path = self._find_cached_dataset()

            if dataset_path and dataset_path.exists():
                print(f"✅ Found cached dataset: {dataset_path}")
            else:
                # Загружаем с Kaggle
                print("📥 Downloading dataset from Kaggle...")
                path = kagglehub.dataset_download(self.__dataset_source.kaggle_handle)
                dataset_path = self.__find_dataset_file(Path(path))
                print(f"✅ Dataset downloaded from Kaggle to: {path}")

        cache_version = (
            f"{PREPROCESSING_VERSION}:{self.__dataset_source.cache_variant}:"
            f"max={max_documents or 'all'}:"
            f"demo={DEMO_SENTENCES_VERSION if include_demo_sentences else 'off'}"
        )

        cached_records = self.__repository.load_sentences(
            dataset_path,
            cache_version,
        )
        if cached_records is not None:
            self.__reset_sentence_lists()
            for document_index, original_text, processed_text in cached_records:
                self.__alias_list.append(document_index)
                self.__orig_sent_list.append(original_text)
                self.__processed_sent_list.append(processed_text)
            print(f"✅ Loaded {len(cached_records)} preprocessed sentences from SQLite")
            return

        self.__main_text_list = list(
            self.__dataset_source.iter_texts(dataset_path, max_documents)
        )
        if include_demo_sentences:
            self.__main_text_list.extend(DEMO_SENTENCES)

        nltk.download('stopwords', quiet=True)
        self.__stop_words = set(stopwords.words('english'))
        self.__reset_sentence_lists()
        self.__fill_lists_by_main_text()
        saved_count = self.__repository.save_sentences(
            dataset_path,
            cache_version,
            zip(self.__alias_list, self.__orig_sent_list, self.__processed_sent_list),
            dataset_name=dataset_name or self.__dataset_source.name,
        )
        print(f"✅ Saved {saved_count} preprocessed sentences to SQLite")

    def get_cached_datasets(self):
        return self.__repository.list_datasets()

    def write_processed_text_to_file(self, filename="output.txt"):
        """Сохраняет элементы __processed_text_list в текстовый файл."""

        files_dir = PROJECT_ROOT / "files"
        
        if not files_dir.exists():
            files_dir.mkdir(parents=True)

        filepath = files_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            text_lines = [str(item) for item in self.__processed_sent_list]
            f.write('\n'.join(text_lines))

        print(f"Сохранено {len(self.__processed_sent_list)} строк в {filepath}")

    def load_text(self, text):
        """Эта функция позволяет вместо загрузки датасета, передать строку, которая и будет исходным текстом. Функция по-большей частью нужна для тестирования."""
        self.__main_text_list = text
        nltk.download('stopwords', quiet=True)
        self.__stop_words = set(stopwords.words('english'))
        self.__reset_sentence_lists()
        self.__fill_lists_by_main_text()

    def get_processed_sentences(self) -> list:
        return self.__processed_sent_list

    def get_original_sentences_by_index(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__orig_sent_list[i])
        return sent_list

    def get_processed_sentences_by_index(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__processed_sent_list[i])
        return sent_list

    def set_stopwords():
        return

    def __reset_sentence_lists(self) -> None:
        self.__orig_sent_list = []
        self.__processed_sent_list = []
        self.__alias_list = []

    def _find_cached_dataset(self) -> Path | None:
        """
        Ищет кэшированный CSV файл датасета.
        Проверяет стандартные директории kagglehub.
        """
        # Путь к кэшу kagglehub
        kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets"
        
        if kaggle_cache.exists():
            # Ищем наш CSV (может быть в поддиректории с версией)
            csv_files = list(kaggle_cache.rglob(self.__dataset_source.filename))
            if csv_files:
                # Берём первый найденный (обычно самый последний)
                return csv_files[0]
        
        return None

    def __find_dataset_file(self, dataset_directory: Path) -> Path:
        matches = list(dataset_directory.rglob(self.__dataset_source.filename))
        if not matches:
            raise FileNotFoundError(
                f"{self.__dataset_source.filename} was not found in {dataset_directory}"
            )
        return matches[0]

    def __resolve_max_documents(self, max_documents: int | None) -> int | None:
        if max_documents is not None:
            return max_documents
        if DATASET_MAX_DOCUMENTS:
            configured_limit = int(DATASET_MAX_DOCUMENTS)
            return configured_limit or None
        return self.__dataset_source.default_max_documents

    def __fill_lists_by_main_text(self):
        for i in range(len(self.__main_text_list)):
            text = self.__main_text_list[i]
            if not isinstance(text, str):
                self.__orig_sent_list.append("")
                self.__processed_sent_list.append("")
                self.__alias_list.append(i)
            else:
                sent_list = sent_tokenize(text)
                for sent in sent_list:
                    self.__orig_sent_list.append(sent)
                    proc_sent = self.__preprocess_sent(sent)
                    self.__processed_sent_list.append(proc_sent)
                    self.__alias_list.append(i)

    def __preprocess_sent(self, sent: str):
        sent_without_links = self.__delete_links(sent.lower())

        words = word_tokenize(sent_without_links)
        filtered_words = [word for word in words
                            if word not in self.__stop_words and word.isalnum()]
        return " ".join(filtered_words)

    def __delete_links(self, text):
        """Очищает текст от ссылок."""
        if pd.isna(text):
            return ""

        # Преобразуем в строку
        text = str(text)

        # Удаляем URL (http/https/ftp)
        url_pattern = r'https?://\S+|ftp://\S+'
        text = re.sub(url_pattern, '', text)

        # Удаляем URL без протокола (начинающиеся с www.)
        www_pattern = r'www\.\S+'
        text = re.sub(www_pattern, '', text)

        # Удаляем URL в скобках
        bracket_pattern = r'\(https?://\S+\)|\(www\.\S+\)'
        text = re.sub(bracket_pattern, '', text)

        # Удаляем URL в квадратных скобках
        square_bracket_pattern = r'\[https?://\S+\]|\[www\.\S+\]'
        text = re.sub(square_bracket_pattern, '', text)

        # Удаляем URL в угловых скобках
        angle_bracket_pattern = r'<https?://\S+>|<www\.\S+>'
        text = re.sub(angle_bracket_pattern, '', text)

        # Удаляем ссылки формата [text](url)
        markdown_pattern = r'\[.*?\]\(https?://\S+\)'
        text = re.sub(markdown_pattern, '', text)

        return text
