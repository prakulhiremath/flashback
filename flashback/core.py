"""Core ``FlashbackFrame`` — a transparent Polars proxy with lineage tracking.

Design principles
-----------------
* **Zero overhead on the happy path** — when lineage tracking is disabled,
  ``FlashbackFrame`` is a thin struct wrapper around a ``polars.DataFrame``.
* **Lazy-first** — operations record metadata eagerly but defer materialisation
  so Polars' query optimiser can still collapse redundant plans.
* **Interoperability** — every Polars DataFrame method is available unchanged;
  the proxy intercepts and wraps the *output*, not the computation.
* **Pandas bridge** — ``to_pandas()`` / ``_from_pandas()`` give seamless
  round-tripping with zero copy when Arrow is available.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import polars as pl

from flashback.dag import LineageDAG

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Intercepted operations — subset of Polars DataFrame API that we track.
# Operations NOT in this set pass through transparently.
# ---------------------------------------------------------------------------

_TRACKED_METHODS: frozenset[str] = frozenset(
    {
        # Selection / projection
        "select",
        "with_columns",
        "rename",
        "drop",
        "drop_nulls",
        # Filtering / slicing
        "filter",
        "head",
        "tail",
        "slice",
        "limit",
        "sample",
        # Sorting
        "sort",
        # Grouping / aggregation
        "group_by",
        "rolling",
        "groupby_dynamic",
        # Joining
        "join",
        "join_where",
        # Reshaping
        "melt",
        "unpivot",
        "pivot",
        "transpose",
        "explode",
        "unnest",
        # Type casting
        "cast",
        # Time-series helpers
        "upsample",
        "shift",
        # Lazy round-trip
        "lazy",
        "collect",
        # Pandas bridge
        "to_pandas",
    }
)


def _make_proxy_method(name: str) -> Any:
    """Factory: build a proxy method that records the call then delegates."""

    @functools.wraps(getattr(pl.DataFrame, name, lambda *a, **kw: None))
    def _method(self: "FlashbackFrame", *args: Any, **kwargs: Any) -> Any:
        # Materialise if currently holding a LazyFrame.
        self._ensure_materialised()

        # Call the underlying Polars method.
        raw_result = getattr(self._df, name)(*args, **kwargs)

        # If the result is a DataFrame (or LazyFrame that we collect), wrap it.
        if isinstance(raw_result, pl.DataFrame):
            op_kwargs = _serialise_args(args, kwargs)
            return self._record_and_wrap(
                data=raw_result,
                op_name=name,
                op_kwargs=op_kwargs,
            )

        if isinstance(raw_result, pl.LazyFrame):
            # Collect eagerly so we can checkpoint the state.
            collected = raw_result.collect()
            op_kwargs = _serialise_args(args, kwargs)
            return self._record_and_wrap(
                data=collected,
                op_name=name,
                op_kwargs=op_kwargs,
            )

        # For non-DataFrame return values (scalars, Series, dicts …) pass through.
        return raw_result

    _method.__name__ = name
    return _method


def _serialise_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert ``*args / **kwargs`` to a JSON-safe dict for storage in the DAG."""

    def _safe(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_safe(i) for i in v]
        if isinstance(v, dict):
            return {str(k): _safe(val) for k, val in v.items()}
        if isinstance(v, pl.Expr):
            # Use stable meta-serialisation so the same expression always
            # produces the same string (repr() includes a memory address).
            try:
                return v.meta.serialize(format="json")
            except Exception:  # noqa: BLE001
                return repr(v)
        if isinstance(v, pl.Series):
            return f"<Series name={v.name!r} len={len(v)}>"
        return str(v)

    result: dict[str, Any] = {}
    for i, arg in enumerate(args):
        result[f"arg_{i}"] = _safe(arg)
    for k, v in kwargs.items():
        result[k] = _safe(v)
    return result


# ---------------------------------------------------------------------------
# FlashbackFrame
# ---------------------------------------------------------------------------


class FlashbackFrame:
    """A transparent Polars DataFrame proxy with automatic lineage tracking.

    Every call to a tracked Polars method (``filter``, ``with_columns``,
    ``join``, etc.) is intercepted, recorded in an internal DAG, and the
    result is returned as a new ``FlashbackFrame`` — preserving the full chain
    for time-travel via :func:`flashback.checkout`.

    Creating instances
    ------------------
    Use :func:`flashback.load` rather than instantiating directly::

        import flashback as fb
        df = fb.load("trades.parquet")

    Or wrap an existing Polars / Pandas frame::

        df = FlashbackFrame._from_polars(polars_df, label="raw")
        df = FlashbackFrame._from_pandas(pandas_df, label="raw")

    Accessing the underlying data
    -----------------------------
    The wrapped ``polars.DataFrame`` is accessible via ``df._df`` or via the
    standard ``to_pandas()`` bridge.  All Polars DataFrame attributes that are
    NOT intercepted (e.g. ``df.dtypes``, ``df.columns``, ``df.schema``) are
    forwarded transparently via ``__getattr__``.
    """

    # Inject proxy methods for all tracked operations at class-definition time.
    # This is done *outside* ``__init__`` so that the method objects are shared
    # across instances (memory efficient) and show up in ``dir()``.
    _tracked = _TRACKED_METHODS

    def __init__(self, df: pl.DataFrame, dag: LineageDAG) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_dag", dag)
        object.__setattr__(self, "_lazy", None)  # holds LazyFrame if deferred

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def _from_polars(
        cls,
        df: pl.DataFrame,
        *,
        label: str = "root",
        message: str = "",
    ) -> "FlashbackFrame":
        """Wrap a ``polars.DataFrame``, creating the root DAG node."""
        dag = LineageDAG()
        dag.add_node(
            op_name="load",
            op_kwargs={"label": label},
            data=df,
            parent_ids=[],
            label=label,
            message=message,
        )
        return cls(df=df, dag=dag)

    @classmethod
    def _from_pandas(
        cls,
        df: "pd.DataFrame",
        *,
        label: str = "root",
        message: str = "",
    ) -> "FlashbackFrame":
        """Wrap a ``pandas.DataFrame`` via a zero-copy Arrow conversion."""
        polars_df = pl.from_pandas(df)
        return cls._from_polars(polars_df, label=label, message=message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_materialised(self) -> None:
        """Collect any pending LazyFrame into ``self._df``."""
        lazy = object.__getattribute__(self, "_lazy")
        if lazy is not None:
            object.__setattr__(self, "_df", lazy.collect())
            object.__setattr__(self, "_lazy", None)

    def _record_and_wrap(
        self,
        *,
        data: pl.DataFrame,
        op_name: str,
        op_kwargs: dict[str, Any],
        label: str = "",
        message: str = "",
    ) -> "FlashbackFrame":
        """Record a new DAG node and return a new ``FlashbackFrame``.

        Each call creates a **child DAG** that is a shallow copy of the current
        DAG up to HEAD, then adds the new node.  This ensures that branching
        (two different transforms applied to the same parent frame) produces two
        independent DAGs with no shared mutable state.
        """
        from flashback.dag import LineageDAG

        parent_dag: LineageDAG = object.__getattribute__(self, "_dag")
        parent_head = parent_dag.head

        # Create a child DAG that inherits the full ancestor history.
        child_dag = LineageDAG()
        child_dag._nodes = dict(parent_dag._nodes)  # shallow-copy node registry
        child_dag._label_index = dict(parent_dag._label_index)
        child_dag._head = parent_dag._head  # start from same HEAD

        child_dag.add_node(
            op_name=op_name,
            op_kwargs=op_kwargs,
            data=data,
            parent_ids=[parent_head.node_id] if parent_head is not None else [],
            label=label,
            message=message,
        )
        return FlashbackFrame(df=data, dag=child_dag)

    def tag(self, label: str, *, message: str = "") -> "FlashbackFrame":
        """Convenience method: tag the current state with a human-readable label.

        Equivalent to :func:`flashback.commit`::

            df = df.tag("before-lag", message="Post-filter checkpoint")

        Parameters
        ----------
        label:
            Unique checkpoint name.
        message:
            Optional description.

        Returns
        -------
        FlashbackFrame
            Self (unchanged), for chaining.
        """
        dag = object.__getattribute__(self, "_dag")
        dag.tag_current(label, message=message)
        # Register with the global registry so fb.checkout() can find this label.
        from flashback.registry import _global_registry
        _global_registry.register(self)
        return self

    # ------------------------------------------------------------------
    # Pandas interop
    # ------------------------------------------------------------------

    def to_pandas(self) -> "pd.DataFrame":
        """Return the underlying data as a ``pandas.DataFrame`` (zero-copy via Arrow)."""
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").to_pandas()

    # ------------------------------------------------------------------
    # Time-series sugar
    # ------------------------------------------------------------------

    def lag(self, column: str, n: int = 1, *, alias: str | None = None) -> "FlashbackFrame":
        """Shift *column* by *n* periods (positive = lag into past).

        Convenience wrapper around ``with_columns(pl.col(column).shift(n))``.

        Parameters
        ----------
        column:
            Column name to lag.
        n:
            Number of periods. Positive = look backward.
        alias:
            Output column name. Defaults to ``f"{column}_lag{n}"``.

        Returns
        -------
        FlashbackFrame
        """
        self._ensure_materialised()
        _alias = alias or f"{column}_lag{n}"
        df = object.__getattribute__(self, "_df")
        result = df.with_columns(pl.col(column).shift(n).alias(_alias))
        return self._record_and_wrap(
            data=result,
            op_name="lag",
            op_kwargs={"column": column, "n": n, "alias": _alias},
        )

    def rolling_mean(
        self,
        column: str,
        window: int,
        *,
        alias: str | None = None,
        min_periods: int | None = None,
    ) -> "FlashbackFrame":
        """Rolling mean over *window* periods for *column*.

        Parameters
        ----------
        column:
            Column name to roll.
        window:
            Window size.
        alias:
            Output column name. Defaults to ``f"{column}_rmean{window}"``.
        min_periods:
            Minimum number of non-null values for a result to be returned.

        Returns
        -------
        FlashbackFrame
        """
        self._ensure_materialised()
        _alias = alias or f"{column}_rmean{window}"
        _min_periods = min_periods if min_periods is not None else window
        df = object.__getattribute__(self, "_df")
        result = df.with_columns(
            pl.col(column)
            .rolling_mean(window_size=window, min_periods=_min_periods)
            .alias(_alias)
        )
        return self._record_and_wrap(
            data=result,
            op_name="rolling_mean",
            op_kwargs={
                "column": column,
                "window": window,
                "alias": _alias,
                "min_periods": _min_periods,
            },
        )

    def group_by(self, *by: Any, **kwargs: Any) -> "_TrackedGroupBy":
        """Tracked ``group_by`` — wraps Polars ``GroupBy`` so that ``.agg()`` results
        are automatically recorded in the lineage DAG.
        """
        self._ensure_materialised()
        df = object.__getattribute__(self, "_df")
        raw_gb = df.group_by(*by, **kwargs)
        return _TrackedGroupBy(raw_gb, self, by, kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        self._ensure_materialised()
        df = object.__getattribute__(self, "_df")
        dag = object.__getattribute__(self, "_dag")
        head = dag.head
        label_part = f"  label={head.label!r}" if head and head.label else ""
        op_part = f"  op={head.op_name!r}" if head else ""
        return (
            f"FlashbackFrame [{df.shape[0]} rows × {df.shape[1]} cols]"
            f"{label_part}{op_part}\n"
            + repr(df)
        )

    def __len__(self) -> int:
        self._ensure_materialised()
        return len(object.__getattribute__(self, "_df"))

    def __getitem__(self, key: Any) -> Any:
        self._ensure_materialised()
        result = object.__getattribute__(self, "_df")[key]
        if isinstance(result, pl.DataFrame):
            return self._record_and_wrap(
                data=result,
                op_name="__getitem__",
                op_kwargs={"key": str(key)},
            )
        return result

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, FlashbackFrame):
            return NotImplemented
        self._ensure_materialised()
        other._ensure_materialised()
        return object.__getattribute__(self, "_df").equals(
            object.__getattribute__(other, "_df")
        )

    def __getattr__(self, name: str) -> Any:
        """Forward any unknown attribute access to the underlying Polars DataFrame."""
        # Tracked methods are handled by the class-level descriptors below.
        if name.startswith("_"):
            raise AttributeError(name)

        self._ensure_materialised()
        df = object.__getattribute__(self, "_df")
        attr = getattr(df, name)

        # If it's a callable, wrap it so that DataFrame outputs are tracked.
        if callable(attr) and name in _TRACKED_METHODS:
            @functools.wraps(attr)
            def _tracked_call(*args: Any, **kwargs: Any) -> Any:
                raw = attr(*args, **kwargs)
                if isinstance(raw, pl.DataFrame):
                    return self._record_and_wrap(
                        data=raw,
                        op_name=name,
                        op_kwargs=_serialise_args(args, kwargs),
                    )
                if isinstance(raw, pl.LazyFrame):
                    collected = raw.collect()
                    return self._record_and_wrap(
                        data=collected,
                        op_name=name,
                        op_kwargs=_serialise_args(args, kwargs),
                    )
                return raw

            return _tracked_call

        return attr

    # ------------------------------------------------------------------
    # Properties that mirror the Polars DataFrame API
    # ------------------------------------------------------------------

    @property
    def schema(self) -> pl.Schema:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").schema

    @property
    def columns(self) -> list[str]:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").columns

    @property
    def dtypes(self) -> list[pl.DataType]:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").dtypes

    @property
    def shape(self) -> tuple[int, int]:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").shape

    @property
    def height(self) -> int:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").height

    @property
    def width(self) -> int:
        self._ensure_materialised()
        return object.__getattribute__(self, "_df").width

    # ------------------------------------------------------------------
    # Lineage helpers
    # ------------------------------------------------------------------

    def history(self) -> list[dict[str, Any]]:
        """Return the list of all DAG nodes as dicts (root → head).

        Useful for programmatic inspection or serialisation::

            for step in df.history():
                print(step["op_name"], step["shape"])
        """
        dag = object.__getattribute__(self, "_dag")
        head = dag.head
        if head is None:
            return []
        return [n.to_dict() for n in dag.ancestors(head.node_id)]

    def diff(self, other: "FlashbackFrame") -> pl.DataFrame:
        """Compare this frame with *other* and return rows that differ.

        Uses a hash-join on all shared columns.  Rows present in ``self`` but
        not in ``other`` are labelled ``"added"``; the reverse are ``"removed"``.

        Parameters
        ----------
        other:
            Another ``FlashbackFrame`` (or ``polars.DataFrame`` coerced to one).

        Returns
        -------
        polars.DataFrame
            Differential with a ``_diff`` column of ``"added"`` / ``"removed"``.
        """
        self._ensure_materialised()
        if isinstance(other, FlashbackFrame):
            other._ensure_materialised()
            other_df = object.__getattribute__(other, "_df")
        elif isinstance(other, pl.DataFrame):
            other_df = other
        else:
            msg = f"Cannot diff FlashbackFrame with {type(other)}"
            raise TypeError(msg)

        self_df = object.__getattribute__(self, "_df")

        added = self_df.join(other_df, on=self_df.columns, how="anti").with_columns(
            pl.lit("added").alias("_diff")
        )
        removed = other_df.join(self_df, on=self_df.columns, how="anti").with_columns(
            pl.lit("removed").alias("_diff")
        )
        return pl.concat([added, removed])


# ---------------------------------------------------------------------------
# GroupBy proxy
# ---------------------------------------------------------------------------


class _TrackedGroupBy:
    """Thin wrapper around ``polars.GroupBy`` that records ``.agg()`` in the DAG."""

    def __init__(
        self,
        raw_gb: Any,
        parent: "FlashbackFrame",
        by: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._raw_gb = raw_gb
        self._parent = parent
        self._by = by
        self._kwargs = kwargs

    def agg(self, *exprs: Any, **named_exprs: Any) -> "FlashbackFrame":
        """Aggregate and record the result in the lineage DAG."""
        result: pl.DataFrame = self._raw_gb.agg(*exprs, **named_exprs)
        op_kwargs = {
            "by": _serialise_args(self._by, self._kwargs),
            "agg": _serialise_args(exprs, named_exprs),
        }
        return self._parent._record_and_wrap(
            data=result,
            op_name="group_by",
            op_kwargs=op_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward any other GroupBy method transparently."""
        return getattr(self._raw_gb, name)
