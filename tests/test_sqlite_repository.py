from pathlib import Path
import sqlite3
from types import SimpleNamespace
from uuid import uuid4
from contextlib import suppress

import pytest

from data.sqlite_repository import SQLiteSentenceRepository
from data.data_storage import DataStorage


@pytest.fixture
def storage_paths():
    files_dir = Path.cwd() / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    source_path = files_dir / f"sqlite-test-{token}.csv"
    database_path = files_dir / f"sqlite-test-{token}.sqlite3"
    yield source_path, database_path
    with suppress(PermissionError):
        source_path.unlink(missing_ok=True)
    with suppress(PermissionError):
        database_path.unlink(missing_ok=True)


def create_source_file(source_path: Path, content: str = "source") -> Path:
    source_path.write_text(content, encoding="utf-8")
    return source_path


def test_save_and_load_sentences(storage_paths):
    source_path, database_path = storage_paths
    create_source_file(source_path)
    repository = SQLiteSentenceRepository(database_path)
    records = [
        (3, "Original one", "original one"),
        (3, "Original two", "original two"),
        (7, "Another document", "another document"),
    ]

    assert repository.save_sentences(source_path, "1", records) == 3
    assert repository.load_sentences(source_path, "1") == records


def test_cache_is_invalidated_when_source_changes(storage_paths):
    source_path, database_path = storage_paths
    create_source_file(source_path)
    repository = SQLiteSentenceRepository(database_path)
    repository.save_sentences(source_path, "1", [(0, "Original", "original")])

    source_path.write_text("changed source content", encoding="utf-8")

    assert repository.load_sentences(source_path, "1") is None


def test_cache_is_invalidated_when_preprocessing_version_changes(storage_paths):
    source_path, database_path = storage_paths
    create_source_file(source_path)
    repository = SQLiteSentenceRepository(database_path)
    repository.save_sentences(source_path, "1", [(0, "Original", "original")])

    assert repository.load_sentences(source_path, "2") is None


def test_saving_new_records_replaces_old_cache(storage_paths):
    source_path, database_path = storage_paths
    create_source_file(source_path)
    repository = SQLiteSentenceRepository(database_path)
    repository.save_sentences(source_path, "1", [(0, "Old", "old")])

    new_records = [(1, "New first", "new first"), (2, "New second", "new second")]
    repository.save_sentences(source_path, "1", new_records)

    assert repository.load_sentences(source_path, "1") == new_records
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 2


def test_data_storage_reuses_preprocessed_sqlite_cache(storage_paths, monkeypatch):
    source_path, database_path = storage_paths
    source_path.write_text('body\n"Buy the car"\n', encoding="utf-8")

    monkeypatch.setattr(DataStorage, "_find_cached_dataset", lambda self: source_path)
    monkeypatch.setattr("data.data_storage.nltk.download", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "data.data_storage.stopwords",
        SimpleNamespace(words=lambda language: ["the"]),
    )
    monkeypatch.setattr("data.data_storage.sent_tokenize", lambda text: [text])
    monkeypatch.setattr("data.data_storage.word_tokenize", lambda text: text.split())

    first_storage = DataStorage(database_path=database_path)
    first_storage.load_data()
    assert first_storage.get_processed_sentences() == ["buy car"]

    monkeypatch.setattr(
        "data.data_storage.pd.read_csv",
        lambda path: (_ for _ in ()).throw(AssertionError("CSV must not be read on cache hit")),
    )

    second_storage = DataStorage(database_path=database_path)
    second_storage.load_data()
    assert second_storage.get_processed_sentences() == ["buy car"]
    assert second_storage.get_original_sentences_by_index([0]) == ["Buy the car"]


def test_repository_keeps_multiple_datasets_isolated(storage_paths):
    first_source, database_path = storage_paths
    second_source = first_source.with_name(f"{first_source.stem}-second.csv")
    create_source_file(first_source, "first source")
    create_source_file(second_source, "second source")
    repository = SQLiteSentenceRepository(database_path)

    try:
        repository.save_sentences(
            first_source,
            "1",
            [(0, "First dataset", "first dataset")],
            dataset_name="First",
        )
        repository.save_sentences(
            second_source,
            "1",
            [(0, "Second dataset", "second dataset")],
            dataset_name="Second",
        )

        assert repository.load_sentences(first_source, "1") == [
            (0, "First dataset", "first dataset")
        ]
        assert repository.load_sentences(second_source, "1") == [
            (0, "Second dataset", "second dataset")
        ]
        assert [dataset.name for dataset in repository.list_datasets()] == ["First", "Second"]

        repository.save_sentences(
            first_source,
            "1",
            [(1, "Updated first", "updated first")],
            dataset_name="First",
        )
        assert repository.load_sentences(second_source, "1") == [
            (0, "Second dataset", "second dataset")
        ]
    finally:
        with suppress(PermissionError):
            second_source.unlink(missing_ok=True)


def test_legacy_database_is_migrated_without_losing_sentences(storage_paths):
    source_path, database_path = storage_paths
    create_source_file(source_path)
    source_stat = source_path.resolve().stat()

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sentences (
                sentence_index INTEGER PRIMARY KEY,
                source_document_index INTEGER NOT NULL,
                original_text TEXT NOT NULL,
                processed_text TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            {
                "source_path": str(source_path.resolve()),
                "source_size": str(source_stat.st_size),
                "source_mtime_ns": str(source_stat.st_mtime_ns),
                "preprocessing_version": "1",
                "status": "ready",
                "sentence_count": "1",
            }.items(),
        )
        connection.execute(
            "INSERT INTO sentences VALUES (?, ?, ?, ?)",
            (0, 4, "Legacy original", "legacy original"),
        )

    repository = SQLiteSentenceRepository(database_path)

    assert repository.load_sentences(source_path, "1") == [
        (4, "Legacy original", "legacy original")
    ]
    assert len(repository.list_datasets()) == 1
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        sentence_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(sentences)")
        }
    assert "metadata" not in tables
    assert "datasets" in tables
    assert "dataset_id" in sentence_columns
