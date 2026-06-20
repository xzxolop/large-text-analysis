from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest

from data.dataset_sources import get_dataset_source


@pytest.fixture
def twitter_csv():
    path = Path.cwd() / "files" / f"twitter-source-{uuid4().hex}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "text,inbound\n"
        '"Customer needs a refund",True\n'
        '"Company response",False\n'
        '"Customer cannot log in",True\n'
        '"",True\n',
        encoding="utf-8",
    )
    yield path
    with suppress(PermissionError):
        path.unlink(missing_ok=True)


def test_twitter_source_returns_only_customer_messages(twitter_csv):
    source = get_dataset_source("twitter_support")

    assert list(source.iter_texts(twitter_csv, max_documents=None)) == [
        "Customer needs a refund",
        "Customer cannot log in",
    ]


def test_twitter_source_respects_document_limit(twitter_csv):
    source = get_dataset_source("twitter_support")

    assert list(source.iter_texts(twitter_csv, max_documents=1)) == [
        "Customer needs a refund"
    ]


def test_unknown_dataset_has_clear_error():
    with pytest.raises(ValueError, match="Available datasets: reddit, twitter_support"):
        get_dataset_source("unknown")
