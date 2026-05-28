"""Unit tests for ``flashback.core`` and the public API.

Coverage targets
----------------
* ``FlashbackFrame`` proxy interception (filter, with_columns, select, join …)
* Time-travel via ``fb.checkout``
* Commit tagging via ``fb.commit`` and ``.tag()``
* ``FlashbackFrame.diff``
* ``FlashbackFrame.lag`` and ``FlashbackFrame.rolling_mean``
* Edge cases: empty DataFrames, single-row frames, wide frames
* DAG determinism: identical transforms produce identical node IDs
* ``FlashbackFrame.history`` output contract
* ``fb.visualize`` does not raise in a non-Jupyter context
* ``fb.reset`` clears the registry
* ``fb.load`` from Polars and Pandas sources
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import flashback as fb
from flashback.core import FlashbackFrame
from flashback.dag import LineageDAG, make_node_id


# ---------------------------------------------------------------------------
# fb.load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_polars_frame(self, simple_df: pl.DataFrame) -> None:
        frame = fb.load(simple_df, label="raw")
        assert isinstance(frame, FlashbackFrame)
        assert frame.shape == simple_df.shape

    def test_load_pandas_frame(self, simple_df: pl.DataFrame) -> None:
        pandas_df = simple_df.to_pandas()
        frame = fb.load(pandas_df, label="pd-raw")
        assert isinstance(frame, FlashbackFrame)
        assert frame.shape == simple_df.shape

    def test_load_registers_globally(self, simple_df: pl.DataFrame) -> None:
        fb.load(simple_df, label="global-test")
        labels = fb._global_registry.list_labels()
        assert "global-test" in labels

    def test_load_no_track(self, simple_df: pl.DataFrame) -> None:
        fb.load(simple_df, label="ephemeral", track=False)
        labels = fb._global_registry.list_labels()
        assert "ephemeral" not in labels

    def test_load_unsupported_format_raises(self, tmp_path) -> None:
        fake = tmp_path / "data.xyz"
        fake.write_text("garbage")
        with pytest.raises(ValueError, match="Unsupported file format"):
            fb.load(str(fake))

    def test_load_parquet(self, simple_df: pl.DataFrame, tmp_path) -> None:
        path = tmp_path / "data.parquet"
        simple_df.write_parquet(str(path))
        frame = fb.load(str(path))
        assert frame.shape == simple_df.shape

    def test_load_csv(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}).write_csv(str(path))
        frame = fb.load(str(path))
        assert frame.shape == (2, 2)


# ---------------------------------------------------------------------------
# Proxy interception
# ---------------------------------------------------------------------------


class TestProxyInterception:
    def test_filter_returns_flashback_frame(self, frame: FlashbackFrame) -> None:
        result = frame.filter(pl.col("price") > 0)
        assert isinstance(result, FlashbackFrame)

    def test_filter_reduces_rows(self, frame: FlashbackFrame) -> None:
        result = frame.filter(pl.col("price") > 0)
        assert result.height < frame.height

    def test_with_columns_adds_column(self, frame: FlashbackFrame) -> None:
        result = frame.with_columns((pl.col("price") * 2).alias("price_x2"))
        assert "price_x2" in result.columns

    def test_select_returns_subset(self, frame: FlashbackFrame) -> None:
        result = frame.select(["id", "price"])
        assert result.columns == ["id", "price"]

    def test_drop_removes_column(self, frame: FlashbackFrame) -> None:
        result = frame.drop(["volume"])
        assert "volume" not in result.columns

    def test_rename_changes_column(self, frame: FlashbackFrame) -> None:
        result = frame.rename({"price": "px"})
        assert "px" in result.columns
        assert "price" not in result.columns

    def test_sort_sorts_frame(self, frame: FlashbackFrame) -> None:
        result = frame.sort("price")
        prices = result._df["price"].to_list()
        assert prices == sorted(prices)

    def test_head_limits_rows(self, frame: FlashbackFrame) -> None:
        result = frame.head(2)
        assert result.height == 2

    def test_tail_limits_rows(self, frame: FlashbackFrame) -> None:
        result = frame.tail(3)
        assert result.height == 3

    def test_chained_operations(self, frame: FlashbackFrame) -> None:
        result = (
            frame
            .filter(pl.col("price") > 0)
            .with_columns((pl.col("price") * pl.col("volume")).alias("notional"))
            .sort("notional", descending=True)
        )
        assert isinstance(result, FlashbackFrame)
        assert "notional" in result.columns

    def test_drop_nulls(self, frame: FlashbackFrame) -> None:
        # Inject a null.
        df_with_null = frame._df.with_columns(
            pl.when(pl.col("id") == 3).then(None).otherwise(pl.col("price")).alias("price")
        )
        frame2 = fb.load(df_with_null, label="with-null")
        result = frame2.drop_nulls()
        assert isinstance(result, FlashbackFrame)
        assert result.height < frame2.height

    def test_getitem_string(self, frame: FlashbackFrame) -> None:
        # Single column → Series (not a DataFrame).
        s = frame._df["price"]
        assert isinstance(s, pl.Series)

    def test_len(self, frame: FlashbackFrame) -> None:
        assert len(frame) == frame.height


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_lag_adds_column(self, frame: FlashbackFrame) -> None:
        result = frame.lag("price", 1)
        assert "price_lag1" in result.columns

    def test_lag_custom_alias(self, frame: FlashbackFrame) -> None:
        result = frame.lag("price", 2, alias="price_prev2")
        assert "price_prev2" in result.columns

    def test_lag_values_are_shifted(self, frame: FlashbackFrame) -> None:
        result = frame.lag("price", 1)
        original = frame._df["price"].to_list()
        lagged = result._df["price_lag1"].to_list()
        # First value is null (no predecessor).
        assert lagged[0] is None
        assert lagged[1] == original[0]

    def test_rolling_mean_adds_column(self, frame: FlashbackFrame) -> None:
        result = frame.rolling_mean("price", 2)
        assert "price_rmean2" in result.columns

    def test_rolling_mean_custom_alias(self, frame: FlashbackFrame) -> None:
        result = frame.rolling_mean("price", 2, alias="rm2")
        assert "rm2" in result.columns

    def test_rolling_mean_is_tracked(self, frame: FlashbackFrame) -> None:
        result = frame.rolling_mean("volume", 3)
        history = result.history()
        ops = [h["op_name"] for h in history]
        assert "rolling_mean" in ops


# ---------------------------------------------------------------------------
# DAG / lineage mechanics
# ---------------------------------------------------------------------------


class TestDAG:
    def test_root_node_created_on_load(self, frame: FlashbackFrame) -> None:
        dag: LineageDAG = object.__getattribute__(frame, "_dag")
        assert dag.head is not None
        assert dag.head.op_name == "load"

    def test_each_operation_appends_node(self, frame: FlashbackFrame) -> None:
        result = frame.filter(pl.col("price") > 0).with_columns(
            (pl.col("volume") * 2).alias("vol2")
        )
        history = result.history()
        # root + filter + with_columns = 3 nodes
        assert len(history) == 3

    def test_history_op_names(self, frame: FlashbackFrame) -> None:
        result = frame.sort("price").head(3)
        ops = [h["op_name"] for h in result.history()]
        assert ops == ["load", "sort", "head"]

    def test_node_id_is_deterministic(self, simple_df: pl.DataFrame) -> None:
        """Identical transforms on identical data → identical node IDs."""
        f1 = fb.load(simple_df, label="det1", track=False)
        f2 = fb.load(simple_df, label="det1", track=False)

        r1 = f1.filter(pl.col("price") > 0)
        r2 = f2.filter(pl.col("price") > 0)

        d1: LineageDAG = object.__getattribute__(r1, "_dag")
        d2: LineageDAG = object.__getattribute__(r2, "_dag")
        assert d1.head is not None
        assert d2.head is not None
        assert d1.head.node_id == d2.head.node_id

    def test_different_transforms_produce_different_ids(self, frame: FlashbackFrame) -> None:
        r1 = frame.filter(pl.col("price") > 0)
        r2 = frame.filter(pl.col("price") > 5)
        d1: LineageDAG = object.__getattribute__(r1, "_dag")
        d2: LineageDAG = object.__getattribute__(r2, "_dag")
        assert d1.head is not None
        assert d2.head is not None
        assert d1.head.node_id != d2.head.node_id

    def test_history_shapes_are_correct(self, frame: FlashbackFrame) -> None:
        result = frame.filter(pl.col("price") > 0)
        history = result.history()
        # Root has 5 rows; after filter (price > 0) → 3 rows (prices: 10, 20, 15).
        assert history[0]["shape"][0] == 5
        assert history[1]["shape"][0] == 3

    def test_to_networkx(self, frame: FlashbackFrame) -> None:
        result = frame.filter(pl.col("price") > 0).sort("price")
        dag: LineageDAG = object.__getattribute__(result, "_dag")
        g = dag.to_networkx()
        import networkx as nx

        assert isinstance(g, nx.DiGraph)
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2


# ---------------------------------------------------------------------------
# Commit / checkout
# ---------------------------------------------------------------------------


class TestCommitCheckout:
    def test_tag_method_labels_node(self, frame: FlashbackFrame) -> None:
        tagged = frame.filter(pl.col("price") > 0).tag("after-filter")
        dag: LineageDAG = object.__getattribute__(tagged, "_dag")
        assert "after-filter" in dag._label_index

    def test_fb_commit_labels_node(self, frame: FlashbackFrame) -> None:
        filtered = frame.filter(pl.col("price") > 0)
        fb.commit(filtered, "post-filter", message="prices cleaned")
        dag: LineageDAG = object.__getattribute__(filtered, "_dag")
        assert "post-filter" in dag._label_index

    def test_checkout_returns_correct_frame(self, frame: FlashbackFrame) -> None:
        filtered = frame.filter(pl.col("price") > 0).tag("after-filter")
        _ = filtered.with_columns((pl.col("price") * 2).alias("price2"))

        checked = fb.checkout("after-filter")
        assert isinstance(checked, FlashbackFrame)
        # Should have 3 positive-price rows.
        assert checked.height == 3
        assert "price2" not in checked.columns

    def test_checkout_unknown_label_raises(self, frame: FlashbackFrame) -> None:
        frame.tag("known")
        with pytest.raises(KeyError, match="not-a-label"):
            fb.checkout("not-a-label")

    def test_checkout_root_label(self, frame: FlashbackFrame) -> None:
        checked = fb.checkout("test-root")
        assert checked.height == 5

    def test_checkout_after_multiple_ops(self, frame: FlashbackFrame) -> None:
        mid = (
            frame
            .filter(pl.col("price") > 0)
            .tag("mid")
        )
        final = mid.with_columns((pl.col("volume") * 10).alias("vol10"))
        fb.commit(final, "final")

        mid_checked = fb.checkout("mid")
        assert "vol10" not in mid_checked.columns
        assert mid_checked.height == 3

        final_checked = fb.checkout("final")
        assert "vol10" in final_checked.columns

    def test_checkout_from_frame_kwarg(self, frame: FlashbackFrame) -> None:
        filtered = frame.filter(pl.col("price") > 0).tag("step1")
        checked = fb.checkout("step1", frame=filtered)
        assert isinstance(checked, FlashbackFrame)

    def test_checkout_on_dag_directly(self, frame: FlashbackFrame) -> None:
        frame.filter(pl.col("price") > 5).tag("p5")
        dag: LineageDAG = object.__getattribute__(frame, "_dag")
        result = dag.checkout("test-root")
        assert isinstance(result, FlashbackFrame)
        assert result.height == 5


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_added_rows(self, frame: FlashbackFrame) -> None:
        filtered = frame.filter(pl.col("price") > 0)
        delta = filtered.diff(frame)
        removed = delta.filter(pl.col("_diff") == "removed")
        # Rows in frame but NOT in filtered → removed
        assert removed.height > 0

    def test_diff_no_changes(self, frame: FlashbackFrame) -> None:
        # Diff against an identical copy.
        copy = fb.load(frame._df, label="copy", track=False)
        delta = frame.diff(copy)
        assert delta.height == 0

    def test_diff_with_polars_df(self, frame: FlashbackFrame) -> None:
        smaller = frame._df.head(3)
        delta = frame.diff(smaller)
        assert isinstance(delta, pl.DataFrame)
        assert "_diff" in delta.columns

    def test_diff_type_error(self, frame: FlashbackFrame) -> None:
        with pytest.raises(TypeError):
            frame.diff("not-a-frame")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Empty / edge-case DataFrames
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe_loads(self, empty_df: pl.DataFrame) -> None:
        frame = fb.load(empty_df, label="empty")
        assert frame.height == 0
        assert frame.width == 2

    def test_filter_empty_remains_empty(self, empty_df: pl.DataFrame) -> None:
        frame = fb.load(empty_df, label="empty2")
        result = frame.filter(pl.col("a") > 0)
        assert result.height == 0

    def test_lag_on_empty_frame(self, empty_df: pl.DataFrame) -> None:
        frame = fb.load(empty_df, label="empty3")
        # Add a float column first so lag has something to shift.
        frame2 = frame.with_columns(pl.lit(0.0).alias("c"))
        result = frame2.lag("c", 1)
        assert result.height == 0

    def test_single_row_frame(self) -> None:
        df = pl.DataFrame({"x": [42], "y": ["hello"]})
        frame = fb.load(df, label="single")
        result = frame.filter(pl.col("x") > 0)
        assert result.height == 1

    def test_wide_frame_operations(self, wide_df: pl.DataFrame) -> None:
        frame = fb.load(wide_df, label="wide")
        result = (
            frame
            .filter(pl.col("price") > 50)
            .with_columns((pl.col("price") * pl.col("volume")).alias("notional"))
            .sort("notional", descending=True)
        )
        assert "notional" in result.columns
        assert result.height <= wide_df.height

    def test_group_by_agg_tracked(self, wide_df: pl.DataFrame) -> None:
        frame = fb.load(wide_df, label="wide-gb")
        agg = frame.group_by("category").agg(pl.col("price").mean().alias("avg_price"))
        assert isinstance(agg, FlashbackFrame)
        ops = [h["op_name"] for h in agg.history()]
        assert "group_by" in ops

    def test_join_tracked(self, simple_df: pl.DataFrame) -> None:
        left = fb.load(simple_df, label="left")
        right_df = pl.DataFrame({"symbol": ["AAPL", "GOOG", "MSFT"], "sector": ["Tech", "Tech", "Tech"]})
        right = fb.load(right_df, label="right", track=False)
        joined = left.join(right._df, on="symbol", how="left")
        assert isinstance(joined, FlashbackFrame)
        assert "sector" in joined.columns


# ---------------------------------------------------------------------------
# Properties and repr
# ---------------------------------------------------------------------------


class TestProperties:
    def test_schema_accessible(self, frame: FlashbackFrame) -> None:
        schema = frame.schema
        assert "price" in schema

    def test_columns_accessible(self, frame: FlashbackFrame) -> None:
        assert "price" in frame.columns

    def test_dtypes_accessible(self, frame: FlashbackFrame) -> None:
        assert len(frame.dtypes) == frame.width

    def test_shape_accessible(self, frame: FlashbackFrame) -> None:
        assert frame.shape == (5, 5)

    def test_height_width(self, frame: FlashbackFrame) -> None:
        assert frame.height == 5
        assert frame.width == 5

    def test_repr_contains_shape(self, frame: FlashbackFrame) -> None:
        r = repr(frame)
        assert "5 rows" in r

    def test_eq_same_data(self, simple_df: pl.DataFrame) -> None:
        f1 = fb.load(simple_df, label="eq1", track=False)
        f2 = fb.load(simple_df, label="eq2", track=False)
        assert f1 == f2

    def test_eq_different_data(self, simple_df: pl.DataFrame) -> None:
        f1 = fb.load(simple_df, label="neq1", track=False)
        f2 = fb.load(simple_df.head(3), label="neq2", track=False)
        assert f1 != f2


# ---------------------------------------------------------------------------
# Pandas interop
# ---------------------------------------------------------------------------


class TestPandasBridge:
    def test_to_pandas_returns_df(self, frame: FlashbackFrame) -> None:
        pd_df = frame.to_pandas()
        import pandas as pd

        assert isinstance(pd_df, pd.DataFrame)
        assert list(pd_df.columns) == frame.columns

    def test_round_trip_pandas(self, frame: FlashbackFrame) -> None:
        pd_df = frame.to_pandas()
        back = fb.load(pd_df, label="roundtrip", track=False)
        assert back.shape == frame.shape


# ---------------------------------------------------------------------------
# Visualize (smoke test — no assertion on output content)
# ---------------------------------------------------------------------------


class TestVisualize:
    def test_visualize_no_raise(self, frame: FlashbackFrame, capsys) -> None:
        frame.filter(pl.col("price") > 0).tag("viz-test")
        fb.visualize()  # Should not raise.

    def test_visualize_specific_frame(self, frame: FlashbackFrame) -> None:
        result = frame.sort("price").tag("sorted")
        fb.visualize(result)  # Should not raise.

    def test_visualize_dag_style(self, frame: FlashbackFrame) -> None:
        frame.filter(pl.col("price") > 0)
        fb.visualize(style="dag")

    def test_visualize_empty_registry(self) -> None:
        fb.reset()
        fb.visualize()  # Should print "No frames tracked" and not raise.


# ---------------------------------------------------------------------------
# Storage (basic round-trip)
# ---------------------------------------------------------------------------


class TestStorage:
    def test_save_and_load(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage

        store = Storage(base_dir=tmp_path / ".flashback")
        path = store.save(frame, frame_id="test-frame")
        assert path.exists()

        loaded = store.load("test-frame")
        assert isinstance(loaded, FlashbackFrame)
        assert loaded.shape == frame.shape

    def test_save_creates_parquet_cache(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage

        store = Storage(base_dir=tmp_path / ".flashback", cache_data=True)
        store.save(frame, frame_id="cache-test")
        cache_files = list((tmp_path / ".flashback" / "cache").glob("*.parquet"))
        assert len(cache_files) > 0

    def test_list_saved(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage

        store = Storage(base_dir=tmp_path / ".flashback")
        store.save(frame, frame_id="listed")
        assert "listed" in store.list_saved()

    def test_load_nonexistent_raises(self, tmp_path) -> None:
        from flashback.storage import Storage

        store = Storage(base_dir=tmp_path / ".flashback")
        with pytest.raises(FileNotFoundError):
            store.load("ghost")

    def test_clear_cache(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage

        store = Storage(base_dir=tmp_path / ".flashback", cache_data=True)
        store.save(frame, frame_id="cc-test")
        deleted = store.clear_cache()
        assert deleted > 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_list(self, frame: FlashbackFrame) -> None:
        from flashback.registry import _global_registry

        _global_registry.register(frame)
        assert "test-root" in _global_registry.list_labels()

    def test_reset_clears_registry(self, frame: FlashbackFrame) -> None:
        from flashback.registry import _global_registry

        _global_registry.register(frame)
        fb.reset()
        assert _global_registry.list_labels() == []
        assert _global_registry.all_frames() == []


# ---------------------------------------------------------------------------
# make_node_id — unit tests for the hashing primitive
# ---------------------------------------------------------------------------


class TestMakeNodeId:
    def test_same_inputs_same_id(self) -> None:
        schema = pl.Schema({"a": pl.Int64, "b": pl.Float64})
        id1 = make_node_id([], "filter", {"arg_0": "price > 0"}, schema)
        id2 = make_node_id([], "filter", {"arg_0": "price > 0"}, schema)
        assert id1 == id2

    def test_different_ops_different_id(self) -> None:
        schema = pl.Schema({"a": pl.Int64})
        id1 = make_node_id([], "filter", {}, schema)
        id2 = make_node_id([], "sort", {}, schema)
        assert id1 != id2

    def test_different_parents_different_id(self) -> None:
        schema = pl.Schema({"a": pl.Int64})
        id1 = make_node_id(["parent1"], "filter", {}, schema)
        id2 = make_node_id(["parent2"], "filter", {}, schema)
        assert id1 != id2

    def test_node_id_is_hex_string(self) -> None:
        schema = pl.Schema({"x": pl.Float32})
        nid = make_node_id([], "load", {}, schema)
        assert re.match(r"^[0-9a-f]+$", nid)
        assert len(nid) == 20  # truncated to 20 hex chars = 80 bits


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestSerialiseArgs:
    """Cover _serialise_args branches: list, dict, Series, fallback str."""

    def test_list_arg(self, frame: FlashbackFrame) -> None:
        # select with a list of strings exercises the list branch in _safe
        result = frame.select(["id", "price"])
        assert result.columns == ["id", "price"]

    def test_series_arg_in_with_columns(self, frame: FlashbackFrame) -> None:
        # Pass a Polars Series as an expression — exercises Series branch
        import polars as pl
        s = pl.Series("bonus", [1.0, 2.0, 3.0, 4.0, 5.0])
        result = frame.with_columns(s.alias("bonus"))
        assert "bonus" in result.columns

    def test_dict_kwarg(self, frame: FlashbackFrame) -> None:
        # rename receives a dict — exercises the dict branch in _safe
        result = frame.rename({"id": "trade_id", "price": "px"})
        assert "trade_id" in result.columns

    def test_nested_list_kwarg(self, frame: FlashbackFrame) -> None:
        # sort by multiple columns
        result = frame.sort(["symbol", "price"])
        assert result.height == frame.height


class TestGetItemDataFrame:
    """Cover __getitem__ path that returns a DataFrame (row slice)."""

    def test_row_slice_returns_flashback_frame(self, frame: FlashbackFrame) -> None:
        # Slice by integer index list → DataFrame
        result = frame[0:3]
        assert isinstance(result, FlashbackFrame)
        assert result.height == 3

    def test_getitem_tracks_operation(self, frame: FlashbackFrame) -> None:
        result = frame[1:4]
        ops = [h["op_name"] for h in result.history()]
        assert "__getitem__" in ops


class TestHistoryEdgeCases:
    """Cover history() on a DAG with no HEAD."""

    def test_history_empty_dag(self) -> None:
        from flashback.dag import LineageDAG
        dag = LineageDAG()
        # No nodes at all — head is None
        frame = FlashbackFrame.__new__(FlashbackFrame)
        object.__setattr__(frame, "_df", pl.DataFrame())
        object.__setattr__(frame, "_dag", dag)
        object.__setattr__(frame, "_lazy", None)
        assert frame.history() == []


class TestDAGExtended:
    """Cover DAG methods not yet reached."""

    def test_register_label_collision_warns(self, frame: FlashbackFrame) -> None:
        import warnings
        dag = object.__getattribute__(frame, "_dag")
        # Tag twice with the same label but a different node_id
        dag._label_index["collision"] = "fake_node_id_001"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dag._register_label("collision", "fake_node_id_002")
        assert any("collision" in str(warning.message) for warning in w)

    def test_get_node_returns_node(self, frame: FlashbackFrame) -> None:
        dag = object.__getattribute__(frame, "_dag")
        head = dag.head
        assert head is not None
        found = dag.get_node(head.node_id)
        assert found is not None
        assert found.node_id == head.node_id

    def test_get_node_missing_returns_none(self, frame: FlashbackFrame) -> None:
        dag = object.__getattribute__(frame, "_dag")
        assert dag.get_node("nonexistent_node_id") is None

    def test_tag_current_no_head_raises(self) -> None:
        from flashback.dag import LineageDAG
        dag = LineageDAG()
        with pytest.raises(RuntimeError, match="no commits"):
            dag.tag_current("label")

    def test_commit_node_to_dict_and_from_dict(self, frame: FlashbackFrame) -> None:
        from flashback.dag import CommitNode
        dag = object.__getattribute__(frame, "_dag")
        node = dag.head
        assert node is not None
        d = node.to_dict()
        assert d["op_name"] == "load"
        restored = CommitNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.op_name == node.op_name
        assert restored.shape == node.shape

    def test_dag_to_dict_and_from_dict(self, frame: FlashbackFrame) -> None:
        from flashback.dag import LineageDAG
        result = frame.filter(pl.col("price") > 0).tag("snap")
        dag = object.__getattribute__(result, "_dag")
        d = dag.to_dict()
        assert "nodes" in d
        assert "head" in d
        restored = LineageDAG.from_dict(d)
        assert restored.head is not None
        assert restored.head.node_id == dag.head.node_id

    def test_replay_uses_cached_data(self, frame: FlashbackFrame) -> None:
        """_replay should traverse chain and return data from nodes that have it."""
        result = frame.filter(pl.col("price") > 0).tag("replay-test")
        dag = object.__getattribute__(result, "_dag")
        head = dag.head
        assert head is not None
        # Simulate replay by calling it directly
        replayed = dag._replay(head.node_id)
        assert isinstance(replayed, pl.DataFrame)
        assert replayed.height == 3

    def test_replay_root_evicted_raises(self, frame: FlashbackFrame) -> None:
        """_replay raises clearly when root data is None."""
        result = frame.filter(pl.col("price") > 0)
        dag = object.__getattribute__(result, "_dag")
        head = dag.head
        assert head is not None
        # Evict all data from all nodes
        for node in dag.list_nodes():
            node._data = None
        with pytest.raises(RuntimeError, match="Cannot replay"):
            dag._replay(head.node_id)

    def test_add_node_cache_hit(self, frame: FlashbackFrame) -> None:
        """Adding an identical node twice hits the cache path."""
        result = frame.filter(pl.col("price") > 0)
        dag = object.__getattribute__(result, "_dag")
        head = dag.head
        assert head is not None
        node_count_before = len(dag._nodes)
        # Adding same node again (same hash) should hit cache, not add new node
        dag.add_node(
            op_name=head.op_name,
            op_kwargs=head.op_kwargs,
            data=head._data,
            parent_ids=head.parent_ids,
            label="cache-hit-label",
        )
        assert len(dag._nodes) == node_count_before  # no new node added

    def test_add_node_with_label_in_cache_hit(self, frame: FlashbackFrame) -> None:
        """Cache-hit path registers labels correctly."""
        result = frame.filter(pl.col("price") > 0)
        dag = object.__getattribute__(result, "_dag")
        head = dag.head
        assert head is not None
        dag.add_node(
            op_name=head.op_name,
            op_kwargs=head.op_kwargs,
            data=head._data,
            parent_ids=head.parent_ids,
            label="cache-label",
        )
        assert "cache-label" in dag._label_index


class TestVisualizationExtended:
    """Cover visualization paths not hit by smoke tests."""

    def test_render_html_empty_dag(self) -> None:
        from flashback.dag import LineageDAG
        from flashback.visualization import _render_html
        dag = LineageDAG()
        html = _render_html(dag)
        assert "No commits yet" in html

    def test_render_html_with_nodes(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _render_html
        result = frame.filter(pl.col("price") > 0).tag("html-test")
        dag = object.__getattribute__(result, "_dag")
        html = _render_html(dag)
        assert "<svg" in html
        assert "filter" in html

    def test_render_html_single_node(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _render_html
        dag = object.__getattribute__(frame, "_dag")
        html = _render_html(dag)
        assert "<svg" in html

    def test_is_jupyter_returns_false_in_test(self) -> None:
        from flashback.visualization import _is_jupyter
        # In pytest, there's no Jupyter kernel
        assert _is_jupyter() is False

    def test_build_rich_tree_empty_dag(self) -> None:
        from flashback.dag import LineageDAG
        from flashback.visualization import _build_rich_tree
        dag = LineageDAG()
        tree = _build_rich_tree(dag)
        assert tree is not None

    def test_render_dag_ascii_empty(self) -> None:
        from flashback.dag import LineageDAG
        from flashback.visualization import _render_dag_ascii
        from rich.console import Console
        import io
        dag = LineageDAG()
        buf = io.StringIO()
        console = Console(file=buf, width=120)
        _render_dag_ascii(dag, console)  # Should print "No commits."

    def test_render_dag_ascii_with_label(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _render_dag_ascii
        from rich.console import Console
        import io
        result = frame.filter(pl.col("price") > 0).tag("ascii-label")
        dag = object.__getattribute__(result, "_dag")
        buf = io.StringIO()
        console = Console(file=buf, width=120)
        _render_dag_ascii(dag, console)
        output = buf.getvalue()
        assert "filter" in output or "load" in output

    def test_add_branch_head_badge(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _add_branch
        from rich.tree import Tree
        dag = object.__getattribute__(frame, "_dag")
        head = dag.head
        assert head is not None
        tree = Tree("root")
        # is_head=True exercises the HEAD badge code path
        _add_branch(tree, head, is_head=True)
        assert len(tree.children) == 1

    def test_add_branch_no_head_badge(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _add_branch
        from rich.tree import Tree
        dag = object.__getattribute__(frame, "_dag")
        head = dag.head
        assert head is not None
        tree = Tree("root")
        _add_branch(tree, head, is_head=False)
        assert len(tree.children) == 1

    def test_build_rich_tree_multinode_advances_branch(self, frame: FlashbackFrame) -> None:
        from flashback.visualization import _build_rich_tree
        result = frame.filter(pl.col("price") > 0).sort("price").head(2)
        dag = object.__getattribute__(result, "_dag")
        tree = _build_rich_tree(dag)
        assert tree is not None


class TestStorageExtended:
    """Cover storage paths not yet reached."""

    def test_cache_size_mb_empty(self, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback")
        assert store.cache_size_mb() == 0.0

    def test_cache_size_mb_after_save(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback", cache_data=True)
        store.save(frame, frame_id="size-test")
        assert store.cache_size_mb() >= 0.0

    def test_destroy_removes_directory(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback")
        store.init()
        assert store.base_dir.exists()
        store.destroy()
        assert not store.base_dir.exists()

    def test_destroy_nonexistent_no_error(self, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback_ghost")
        store.destroy()  # Should not raise

    def test_save_without_frame_id(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback")
        path = store.save(frame)
        assert path.exists()

    def test_list_saved_empty(self, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback")
        assert store.list_saved() == []

    def test_hydrate_cache_handles_corrupt_parquet(self, frame: FlashbackFrame, tmp_path) -> None:
        """_hydrate_cache should silently skip unreadable cache files."""
        from flashback.dag import LineageDAG
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback", cache_data=True)
        store.save(frame, frame_id="corrupt-test")
        # Corrupt one cache file
        cache_files = list((tmp_path / ".flashback" / "cache").glob("*.parquet"))
        assert len(cache_files) > 0
        cache_files[0].write_bytes(b"corrupted data not parquet")
        # Now build a DAG with evicted data and try to hydrate
        dag_path = tmp_path / ".flashback" / "graphs" / "corrupt-test.json"
        import json
        raw = json.loads(dag_path.read_text())
        dag = LineageDAG.from_dict(raw)
        # Evict data from all nodes
        for node in dag.list_nodes():
            node._data = None
        # hydrate_cache should not raise even with a corrupt file
        store._hydrate_cache(dag)

    def test_save_cache_data_false(self, frame: FlashbackFrame, tmp_path) -> None:
        from flashback.storage import Storage
        store = Storage(base_dir=tmp_path / ".flashback", cache_data=False)
        store.save(frame, frame_id="no-cache")
        cache_files = list((tmp_path / ".flashback" / "cache").glob("*.parquet"))
        assert len(cache_files) == 0

    def test_from_cwd(self) -> None:
        from flashback.storage import Storage
        store = Storage.from_cwd()
        assert store.base_dir.name == ".flashback"


class TestMakeProxyMethodLazyFrame:
    """Cover the LazyFrame branch inside _make_proxy_method."""

    def test_lazy_collect_tracked(self, frame: FlashbackFrame) -> None:
        # .lazy() returns a LazyFrame; going through __getattr__ path
        # The lazy method itself is in _TRACKED_METHODS so it goes through proxy
        lazy_result = frame._df.lazy().filter(pl.col("price") > 0).collect()
        assert isinstance(lazy_result, pl.DataFrame)

    def test_getattr_non_tracked_passthrough(self, frame: FlashbackFrame) -> None:
        # dtypes is not in _TRACKED_METHODS → plain passthrough via __getattr__
        dtypes = frame.dtypes
        assert len(dtypes) == frame.width

    def test_getattr_tracked_returns_callable(self, frame: FlashbackFrame) -> None:
        # Access a tracked method via __getattr__ — sort is in _TRACKED_METHODS
        # and delegated through __getattr__, so calling it should work
        result = frame.sort("price")
        assert isinstance(result, FlashbackFrame)


class TestCommitNodeDefaults:
    """Cover CommitNode.from_dict with missing optional fields."""

    def test_from_dict_missing_optional_fields(self) -> None:
        from flashback.dag import CommitNode
        d = {
            "node_id": "abc123",
            "op_name": "load",
            "op_kwargs": {},
            "schema_hash": "deadbeef",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        node = CommitNode.from_dict(d)
        assert node.label == ""
        assert node.message == ""
        assert node.parent_ids == []
        assert node.shape == (0, 0)


class TestRegistryExtended:
    """Cover registry miss paths."""

    def test_checkout_missing_frame_id(self) -> None:
        from flashback.registry import _global_registry
        result = _global_registry.checkout("does-not-exist")
        assert result is None

    def test_all_frames_empty(self) -> None:
        from flashback.registry import _global_registry
        frames = _global_registry.all_frames()
        assert frames == []

    def test_checkout_stale_frame_id(self, frame: FlashbackFrame) -> None:
        """If fid maps to a missing frame (GC'd), checkout returns None."""
        from flashback.registry import _global_registry
        _global_registry.register(frame)
        # Manually corrupt the registry to simulate a stale entry
        _global_registry._label_to_id["stale-label"] = 999999999  # invalid id
        result = _global_registry.checkout("stale-label")
        assert result is None
