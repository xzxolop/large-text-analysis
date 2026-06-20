"""Dataset-specific adapters that expose a common stream of text documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class DatasetSource:
    key: str
    name: str
    kaggle_handle: str
    filename: str
    text_column: str
    default_max_documents: int | None = None
    customer_messages_only: bool = False

    @property
    def cache_variant(self) -> str:
        direction = "customer" if self.customer_messages_only else "all"
        return f"{self.key}:{direction}"

    def iter_texts(
        self,
        dataset_path: Path,
        max_documents: int | None,
        chunk_size: int = 50_000,
    ) -> Iterator[str]:
        usecols = [self.text_column]
        if self.customer_messages_only:
            usecols.append("inbound")

        emitted = 0
        for chunk in pd.read_csv(dataset_path, usecols=usecols, chunksize=chunk_size):
            if self.customer_messages_only:
                inbound = chunk["inbound"].astype(str).str.lower().eq("true")
                chunk = chunk.loc[inbound]

            for text in chunk[self.text_column].dropna():
                if isinstance(text, str) and text.strip():
                    yield text
                    emitted += 1
                    if max_documents is not None and emitted >= max_documents:
                        return


DATASET_SOURCES = {
    "reddit": DatasetSource(
        key="reddit",
        name="Reddit comments",
        kaggle_handle="pavellexyr/the-reddit-dataset-dataset",
        filename="the-reddit-dataset-dataset-comments.csv",
        text_column="body",
    ),
    "twitter_support": DatasetSource(
        key="twitter_support",
        name="Customer Support on Twitter",
        kaggle_handle="thoughtvector/customer-support-on-twitter",
        filename="twcs.csv",
        text_column="text",
        default_max_documents=100_000,
        customer_messages_only=True,
    ),
}


def get_dataset_source(key: str) -> DatasetSource:
    try:
        return DATASET_SOURCES[key]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unknown dataset '{key}'. Available datasets: {available}") from error
