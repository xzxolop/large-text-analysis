"""
Конфигурация проекта.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Путь к корню проекта (на уровень выше src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Папка для файлов из .env (по умолчанию 'files')
FILES_DIR = os.getenv("FILES_DIR", "files")

# Источник данных, используемый при запуске приложения.
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "reddit")
#ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "twitter_support")

# Пустое значение использует безопасный лимит, заданный адаптером датасета.
DATASET_MAX_DOCUMENTS = os.getenv("DATASET_MAX_DOCUMENTS")

# Добавлять синтетические предложения для демонстрации поиска спроса.
INCLUDE_DEMO_SENTENCES = os.getenv(
    "INCLUDE_DEMO_SENTENCES",
    "true",
).lower() in {"1", "true", "yes"}
