"""Disk persistence for flashback lineage graphs.

The ``.flashback/`` directory layout
--------------------------------------
::

    .flashback/
    ├── config.json          # global settings
    ├── graphs/
    │   └── <frame_id>.json  # serialised LineageDAG per frame
    └── cache/
        └── <node_id>.parquet  # optional materialised data cache

Design
------
* Persistence is **opt-in** — call :func:`Storage.save` explicitly.
* Hashing uses ``xxhash`` (XX3/64) for speed-critical path; SHA-256 is
  reserved for the deterministic node IDs in :mod:`flashback.dag`.
* The ``cache/`` sub-directory stores materialised Parquet snapshots keyed by
  ``node_id``.  Reads first check in-memory DAG data; on miss they hit the
  Parquet cache; on a double miss they replay from root.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from flashback.core import FlashbackFrame
    from flashback.dag import LineageDAG

_DEFAULT_DIR = pathlib.Path(".flashback")

_CONFIG_DEFAULTS: dict[str, Any] = {
    "version": "0.1.0",
    "cache_data": True,  # whether to persist Parquet snapshots
    "max_cache_mb": 512,
}


class Storage:
    """Handle serialisation and deserialisation of :class:`~flashback.dag.LineageDAG`.

    Parameters
    ----------
    base_dir:
        Root directory for all flashback state.  Defaults to ``.flashback/``
        in the current working directory.
    cache_data:
        Whether to write materialised Parquet files to ``cache/``.  Disable
        for large datasets or when disk is constrained.
    """

    def __init__(
        self,
        base_dir: pathlib.Path | str = _DEFAULT_DIR,
        *,
        cache_data: bool = True,
    ) -> None:
        self.base_dir = pathlib.Path(base_dir)
        self.cache_data = cache_data
        self._graphs_dir = self.base_dir / "graphs"
        self._cache_dir = self.base_dir / "cache"
        self._config_path = self.base_dir / "config.json"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create the ``.flashback/`` directory structure if it does not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._graphs_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if not self._config_path.exists():
            cfg = dict(_CONFIG_DEFAULTS)
            cfg["cache_data"] = self.cache_data
            self._config_path.write_text(
                json.dumps(cfg, indent=2),
                encoding="utf-8",
            )

    @classmethod
    def from_cwd(cls) -> "Storage":
        """Create a ``Storage`` rooted at ``.flashback/`` in the current directory."""
        return cls(base_dir=_DEFAULT_DIR)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, frame: "FlashbackFrame", *, frame_id: str | None = None) -> pathlib.Path:
        """Persist the DAG of *frame* to disk.

        Parameters
        ----------
        frame:
            The frame whose lineage to persist.
        frame_id:
            A unique identifier for this frame (used as the filename stem).
            If ``None``, derived from the root node's ``node_id``.

        Returns
        -------
        pathlib.Path
            Path to the written ``.json`` graph file.
        """
        self.init()
        dag: "LineageDAG" = object.__getattribute__(frame, "_dag")

        _frame_id: str
        if frame_id is not None:
            _frame_id = frame_id
        elif dag.head is not None:
            # Use the root node id (stable across runs with the same source data).
            root = dag.ancestors(dag.head.node_id)[0]
            _frame_id = root.node_id
        else:
            import uuid

            _frame_id = uuid.uuid4().hex

        graph_path = self._graphs_dir / f"{_frame_id}.json"
        graph_path.write_text(
            json.dumps(dag.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        if self.cache_data:
            self._write_cache(dag)

        return graph_path

    def _write_cache(self, dag: "LineageDAG") -> None:
        """Write Parquet snapshots for all nodes that have materialised data."""
        for node in dag.list_nodes():
            if node._data is not None:
                cache_path = self._cache_dir / f"{node.node_id}.parquet"
                if not cache_path.exists():
                    node._data.write_parquet(str(cache_path))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, frame_id: str) -> "FlashbackFrame":
        """Load a ``FlashbackFrame`` from a persisted graph file.

        Parameters
        ----------
        frame_id:
            The filename stem (i.e. the ``node_id`` of the root node) passed
            to :meth:`save`.

        Returns
        -------
        FlashbackFrame
        """
        from flashback.core import FlashbackFrame
        from flashback.dag import LineageDAG

        graph_path = self._graphs_dir / f"{frame_id}.json"
        if not graph_path.exists():
            available = [p.stem for p in self._graphs_dir.glob("*.json")]
            msg = f"No saved graph '{frame_id}' in {self._graphs_dir}. Available: {available}"
            raise FileNotFoundError(msg)

        raw = json.loads(graph_path.read_text(encoding="utf-8"))
        dag = LineageDAG.from_dict(raw)

        # Attempt to populate node._data from the Parquet cache.
        self._hydrate_cache(dag)

        head = dag.head
        if head is None or head._data is None:
            msg = f"Graph '{frame_id}' has no materialisable HEAD node."
            raise RuntimeError(msg)

        return FlashbackFrame(df=head._data, dag=dag)

    def _hydrate_cache(self, dag: "LineageDAG") -> None:
        """Read Parquet cache files back into ``CommitNode._data``."""
        for node in dag.list_nodes():
            cache_path = self._cache_dir / f"{node.node_id}.parquet"
            if node._data is None and cache_path.exists():
                try:
                    node._data = pl.read_parquet(str(cache_path))
                except Exception:  # noqa: BLE001
                    # Non-fatal: data replay will handle it.
                    pass

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_saved(self) -> list[str]:
        """Return the ``frame_id`` stems of all saved graphs."""
        if not self._graphs_dir.exists():
            return []
        return sorted(p.stem for p in self._graphs_dir.glob("*.json"))

    def cache_size_mb(self) -> float:
        """Return total size of the Parquet cache in megabytes."""
        if not self._cache_dir.exists():
            return 0.0
        total = sum(p.stat().st_size for p in self._cache_dir.glob("*.parquet"))
        return round(total / (1024 * 1024), 3)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_cache(self) -> int:
        """Delete all Parquet snapshot files.  Returns the count deleted."""
        if not self._cache_dir.exists():
            return 0
        files = list(self._cache_dir.glob("*.parquet"))
        for f in files:
            f.unlink()
        return len(files)

    def destroy(self) -> None:
        """Recursively delete the entire ``.flashback/`` directory."""
        import shutil

        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
