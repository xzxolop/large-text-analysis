from pathlib import Path
import sqlite3
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
    monkeypatch.setattr("data.data_storage.stopwords.words", lambda language: ["the"])
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
