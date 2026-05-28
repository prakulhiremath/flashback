"""Shared fixtures for the flashback test suite."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

import flashback as fb
from flashback.core import FlashbackFrame


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Ensure every test starts with a clean global registry."""
    fb.reset()
    yield  # type: ignore[misc]
    fb.reset()


@pytest.fixture()
def simple_df() -> pl.DataFrame:
    """A small, well-defined Polars DataFrame for deterministic tests."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "price": [10.0, -5.0, 20.0, 15.0, 0.0],
            "volume": [100, 200, 150, 80, 300],
            "symbol": ["AAPL", "GOOG", "AAPL", "MSFT", "GOOG"],
            "ts": [
                datetime.date(2024, 1, i) for i in range(1, 6)
            ],
        }
    )


@pytest.fixture()
def frame(simple_df: pl.DataFrame) -> FlashbackFrame:
    """A ``FlashbackFrame`` wrapping ``simple_df``."""
    return fb.load(simple_df, label="test-root")


@pytest.fixture()
def empty_df() -> pl.DataFrame:
    return pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Float64)})


@pytest.fixture()
def wide_df() -> pl.DataFrame:
    """A wider DataFrame for aggregation / join tests."""
    import random

    random.seed(42)
    n = 50
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "price": [round(random.uniform(1, 200), 2) for _ in range(n)],
            "volume": [random.randint(10, 1000) for _ in range(n)],
            "category": [random.choice(["A", "B", "C"]) for _ in range(n)],
        }
    )
