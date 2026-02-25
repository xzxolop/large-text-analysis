import os
from pathlib import Path
from typing import Iterable, Tuple


# Путь к корню проекта (на уровень выше src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DataExporter:
    """
    Отвечает за сохранение результатов анализа (пока только TF-IDF) в файл.
    """

    def write_mean_tfidf_to_file(
        self,
        tfidf_list: Iterable[Tuple[str, float]],
        filename: str = "tfidf_results.txt",
        folder_path: str = "files",
    ) -> str:
        """
        Записывает пары (слово, score) в текстовый файл и возвращает путь.
        """
        # Используем абсолютный путь относительно корня проекта
        files_dir = PROJECT_ROOT / folder_path
        
        if not files_dir.exists():
            files_dir.mkdir(parents=True)

        filepath = files_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for word, score in tfidf_list:
                f.write(f"{word}: {score:.6f}\n")

        return str(filepath)
