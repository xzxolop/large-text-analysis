import os
from typing import Iterable, Tuple


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
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filepath = os.path.join(folder_path, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for word, score in tfidf_list:
                f.write(f"{word}: {score:.6f}\n")

        return filepath
