"""
flashback — Git for Datasets.

Time-travel debugging and lineage tracking for pandas/Polars DataFrames.

Public API
----------
    >>> import flashback as fb
    >>> df = fb.load("trades.parquet")
    >>> df = df.filter(fb.col("price") > 0).with_columns(fb.col("price").shift(1).alias("price_lag"))
    >>> df_old = fb.checkout("before-lag")
    >>> fb.visualize()
"""

from __future__ import annotations

from flashback.core import FlashbackFrame
from flashback.registry import _global_registry
from flashback.storage import Storage

__all__ = [
    "FlashbackFrame",
    "checkout",
    "col",
    "commit",
    "load",
    "reset",
    "visualize",
]

__version__ = "0.1.0"
__author__ = "flashback contributors"


def load(
    source: str | "FlashbackFrame" | "polars.DataFrame" | "pandas.DataFrame",  # noqa: F821
    *,
    label: str | None = None,
    track: bool = True,
) -> "FlashbackFrame":
    """Load a DataFrame from a file path or an existing frame, enabling lineage tracking.

    Parameters
    ----------
    source:
        A file path (Parquet, CSV, JSON, NDJSON), a Polars DataFrame, a Pandas
        DataFrame, or another ``FlashbackFrame``.
    label:
        Optional human-readable label for the initial commit node. Defaults to
        the filename stem or ``"root"``.
    track:
        Whether to register the frame with the global registry. Set to
        ``False`` for ephemeral frames you don't want tracked.

    Returns
    -------
    FlashbackFrame
        A proxy frame with lineage tracking enabled.

    Examples
    --------
    >>> import flashback as fb
    >>> df = fb.load("trades.parquet")
    >>> df = fb.load("prices.csv", label="raw-prices")
    """
    import pathlib

    import polars as pl

    if isinstance(source, FlashbackFrame):
        frame = source
    elif isinstance(source, pl.DataFrame):
        frame = FlashbackFrame._from_polars(source, label=label or "root")
    else:
        try:
            import pandas as pd  # type: ignore[import-untyped]

            if isinstance(source, pd.DataFrame):
                frame = FlashbackFrame._from_pandas(source, label=label or "root")
                if track:
                    _global_registry.register(frame)
                return frame
        except ImportError:
            pass

        path = pathlib.Path(source)
        _label = label or path.stem
        suffix = path.suffix.lower()
        loaders = {
            ".parquet": pl.read_parquet,
            ".csv": pl.read_csv,
            ".json": pl.read_json,
            ".ndjson": pl.read_ndjson,
            ".ipc": pl.read_ipc,
            ".arrow": pl.read_ipc,
        }
        loader = loaders.get(suffix)
        if loader is None:
            msg = f"Unsupported file format: '{suffix}'. Supported: {list(loaders)}"
            raise ValueError(msg)
        raw = loader(str(path))
        frame = FlashbackFrame._from_polars(raw, label=_label)

    if track:
        _global_registry.register(frame)
    return frame


def col(name: str) -> "polars.Expr":  # type: ignore[name-defined]  # noqa: F821
    """Alias for ``polars.col`` — use inside flashback transform chains.

    Parameters
    ----------
    name:
        Column name.

    Returns
    -------
    polars.Expr
    """
    import polars as pl

    return pl.col(name)


def checkout(
    label: str,
    *,
    frame: "FlashbackFrame | None" = None,
) -> "FlashbackFrame":
    """Time-travel: return the ``FlashbackFrame`` at a named commit checkpoint.

    Parameters
    ----------
    label:
        The commit label (e.g. ``"before-lag"``). Must match a label previously
        set via :func:`commit` or the ``label`` parameter of a transform.
    frame:
        If provided, search within this frame's lineage graph only. If
        ``None``, searches the global registry.

    Returns
    -------
    FlashbackFrame
        The frame as it existed at that checkpoint, fully materialized.

    Raises
    ------
    KeyError
        If no checkpoint with the given label exists.

    Examples
    --------
    >>> df_original = fb.checkout("raw-prices")
    """
    if frame is not None:
        return frame._dag.checkout(label)

    target = _global_registry.checkout(label)
    if target is None:
        available = _global_registry.list_labels()
        msg = (
            f"No checkpoint found with label '{label}'. "
            f"Available labels: {available}"
        )
        raise KeyError(msg)
    return target


def commit(
    frame: "FlashbackFrame",
    label: str,
    *,
    message: str = "",
) -> "FlashbackFrame":
    """Manually tag the current state of *frame* with a human-readable label.

    This is analogous to ``git tag`` — it pins a name to the current DAG node
    so you can :func:`checkout` to it later.

    Parameters
    ----------
    frame:
        The frame to tag.
    label:
        A unique, human-readable label.
    message:
        Optional description stored alongside the commit metadata.

    Returns
    -------
    FlashbackFrame
        The same frame, unchanged, with the label registered.

    Examples
    --------
    >>> df = fb.commit(df, "before-lag", message="Post-filter, pre-feature-eng")
    """
    frame._dag.tag_current(label, message=message)
    _global_registry.register(frame)
    return frame


def visualize(
    frame: "FlashbackFrame | None" = None,
    *,
    style: str = "tree",
    max_width: int = 120,
) -> None:
    """Render the transformation lineage graph to the terminal (or Jupyter).

    Parameters
    ----------
    frame:
        The frame whose lineage to visualize. If ``None``, renders all frames
        tracked in the global registry.
    style:
        ``"tree"`` (default) for a rich tree view, ``"dag"`` for a compact
        ASCII DAG layout.
    max_width:
        Terminal width cap for rich output.

    Examples
    --------
    >>> fb.visualize()
    >>> fb.visualize(df, style="dag")
    """
    from flashback.visualization import render

    if frame is not None:
        render(frame._dag, style=style, max_width=max_width)
    else:
        frames = _global_registry.all_frames()
        if not frames:
            from rich.console import Console

            Console().print("[dim]No frames currently tracked by flashback.[/dim]")
            return
        for f in frames:
            render(f._dag, style=style, max_width=max_width)


def reset() -> None:
    """Clear the global registry. Useful between experiments or in tests."""
    _global_registry.clear()
