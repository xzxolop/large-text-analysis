"""SQLite persistence for preprocessed dataset sentences."""

from pathlib import Path
import sqlite3
from typing import Iterable, Optional


SentenceRecord = tuple[int, str, str]


class SQLiteSentenceRepository:
    """Stores original and processed sentences between application runs."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sentences (
                    sentence_index INTEGER PRIMARY KEY,
                    source_document_index INTEGER NOT NULL,
                    original_text TEXT NOT NULL,
                    processed_text TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _source_metadata(dataset_path: Path, preprocessing_version: str) -> dict[str, str]:
        resolved_path = dataset_path.resolve()
        stat = resolved_path.stat()
        return {
            "source_path": str(resolved_path),
            "source_size": str(stat.st_size),
            "source_mtime_ns": str(stat.st_mtime_ns),
            "preprocessing_version": preprocessing_version,
        }

    def load_sentences(
        self,
        dataset_path: Path,
        preprocessing_version: str,
    ) -> Optional[list[SentenceRecord]]:
        """Return cached records only when source and preprocessing version match."""
        expected = self._source_metadata(dataset_path, preprocessing_version)

        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("status") != "ready":
                return None
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None

            rows = connection.execute(
                """
                SELECT source_document_index, original_text, processed_text
                FROM sentences
                ORDER BY sentence_index
                """
            ).fetchall()

        if len(rows) != int(metadata.get("sentence_count", "-1")):
            return None
        return [(int(document_index), original, processed) for document_index, original, processed in rows]

    def save_sentences(
        self,
        dataset_path: Path,
        preprocessing_version: str,
        records: Iterable[SentenceRecord],
    ) -> int:
        """Atomically replace the cached dataset and return the saved row count."""
        metadata = self._source_metadata(dataset_path, preprocessing_version)

        with self._connect() as connection:
            connection.execute("DELETE FROM sentences")
            connection.execute("DELETE FROM metadata")
            connection.executemany(
                """
                INSERT INTO sentences (
                    sentence_index,
                    source_document_index,
                    original_text,
                    processed_text
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (sentence_index, document_index, original, processed)
                    for sentence_index, (document_index, original, processed)
                    in enumerate(records)
                ),
            )
            sentence_count = connection.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
            metadata.update({"status": "ready", "sentence_count": str(sentence_count)})
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                metadata.items(),
            )

        return sentence_count
