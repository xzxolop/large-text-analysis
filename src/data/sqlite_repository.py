"""SQLite persistence for preprocessed datasets and their sentences."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, Optional


SentenceRecord = tuple[int, str, str]


@dataclass(frozen=True)
class DatasetInfo:
    id: int
    name: str
    source_path: str
    status: str
    sentence_count: int


class SQLiteSentenceRepository:
    """Stores multiple preprocessed datasets in a single SQLite database."""

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

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _create_current_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_path TEXT NOT NULL UNIQUE,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                preprocessing_version TEXT NOT NULL,
                status TEXT NOT NULL,
                sentence_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sentences (
                dataset_id INTEGER NOT NULL,
                sentence_index INTEGER NOT NULL,
                source_document_index INTEGER NOT NULL,
                original_text TEXT NOT NULL,
                processed_text TEXT NOT NULL,
                PRIMARY KEY (dataset_id, sentence_index),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sentences_dataset
            ON sentences(dataset_id);
            """
        )

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            if self._is_legacy_schema(connection):
                self._migrate_legacy_schema(connection)
            self._create_current_schema(connection)

    def _is_legacy_schema(self, connection: sqlite3.Connection) -> bool:
        if not self._table_exists(connection, "metadata"):
            return False
        if not self._table_exists(connection, "sentences"):
            return False
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(sentences)")
        }
        return "dataset_id" not in columns

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> None:
        """Convert the original single-dataset cache without reprocessing text."""
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        connection.execute("ALTER TABLE sentences RENAME TO sentences_legacy")
        self._create_current_schema(connection)

        source_path = metadata.get("source_path", "legacy-dataset")
        dataset_name = Path(source_path).stem or "Legacy dataset"
        cursor = connection.execute(
            """
            INSERT INTO datasets (
                name,
                source_path,
                source_size,
                source_mtime_ns,
                preprocessing_version,
                status,
                sentence_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                source_path,
                int(metadata.get("source_size", "0")),
                int(metadata.get("source_mtime_ns", "0")),
                metadata.get("preprocessing_version", "1"),
                metadata.get("status", "ready"),
                int(metadata.get("sentence_count", "0")),
            ),
        )
        dataset_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO sentences (
                dataset_id,
                sentence_index,
                source_document_index,
                original_text,
                processed_text
            )
            SELECT ?, sentence_index, source_document_index, original_text, processed_text
            FROM sentences_legacy
            """,
            (dataset_id,),
        )
        connection.execute("DROP TABLE sentences_legacy")
        connection.execute("DROP TABLE metadata")

    @staticmethod
    def _source_metadata(dataset_path: Path, preprocessing_version: str) -> dict[str, object]:
        resolved_path = dataset_path.resolve()
        stat = resolved_path.stat()
        return {
            "source_path": str(resolved_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "preprocessing_version": preprocessing_version,
        }

    def list_datasets(self) -> list[DatasetInfo]:
        """Return every dataset currently stored in the database."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, source_path, status, sentence_count
                FROM datasets
                ORDER BY id
                """
            ).fetchall()
        return [DatasetInfo(*row) for row in rows]

    def load_sentences(
        self,
        dataset_path: Path,
        preprocessing_version: str,
    ) -> Optional[list[SentenceRecord]]:
        """Return cached records only when source and preprocessing version match."""
        expected = self._source_metadata(dataset_path, preprocessing_version)

        with self._connect() as connection:
            dataset = connection.execute(
                """
                SELECT id, source_size, source_mtime_ns,
                       preprocessing_version, status, sentence_count
                FROM datasets
                WHERE source_path = ?
                """,
                (expected["source_path"],),
            ).fetchone()
            if dataset is None:
                return None

            dataset_id, source_size, source_mtime_ns, version, status, sentence_count = dataset
            if status != "ready":
                return None
            if source_size != expected["source_size"]:
                return None
            if source_mtime_ns != expected["source_mtime_ns"]:
                return None
            if version != expected["preprocessing_version"]:
                return None

            rows = connection.execute(
                """
                SELECT source_document_index, original_text, processed_text
                FROM sentences
                WHERE dataset_id = ?
                ORDER BY sentence_index
                """,
                (dataset_id,),
            ).fetchall()

        if len(rows) != sentence_count:
            return None
        return [(int(document_index), original, processed) for document_index, original, processed in rows]

    def save_sentences(
        self,
        dataset_path: Path,
        preprocessing_version: str,
        records: Iterable[SentenceRecord],
        dataset_name: Optional[str] = None,
    ) -> int:
        """Atomically replace one dataset without affecting other datasets."""
        metadata = self._source_metadata(dataset_path, preprocessing_version)
        name = dataset_name or Path(str(metadata["source_path"])).stem

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    name,
                    source_path,
                    source_size,
                    source_mtime_ns,
                    preprocessing_version,
                    status,
                    sentence_count
                ) VALUES (?, ?, ?, ?, ?, 'building', 0)
                ON CONFLICT(source_path) DO UPDATE SET
                    name = excluded.name,
                    source_size = excluded.source_size,
                    source_mtime_ns = excluded.source_mtime_ns,
                    preprocessing_version = excluded.preprocessing_version,
                    status = 'building',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    name,
                    metadata["source_path"],
                    metadata["source_size"],
                    metadata["source_mtime_ns"],
                    metadata["preprocessing_version"],
                ),
            )
            dataset_id = connection.execute(
                "SELECT id FROM datasets WHERE source_path = ?",
                (metadata["source_path"],),
            ).fetchone()[0]

            connection.execute("DELETE FROM sentences WHERE dataset_id = ?", (dataset_id,))
            connection.executemany(
                """
                INSERT INTO sentences (
                    dataset_id,
                    sentence_index,
                    source_document_index,
                    original_text,
                    processed_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (dataset_id, sentence_index, document_index, original, processed)
                    for sentence_index, (document_index, original, processed)
                    in enumerate(records)
                ),
            )
            sentence_count = connection.execute(
                "SELECT COUNT(*) FROM sentences WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()[0]
            connection.execute(
                """
                UPDATE datasets
                SET status = 'ready', sentence_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (sentence_count, dataset_id),
            )

        return sentence_count
