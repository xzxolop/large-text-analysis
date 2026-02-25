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
