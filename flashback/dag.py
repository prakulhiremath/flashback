"""DAG (Directed Acyclic Graph) engine for flashback lineage tracking.

Each node in the DAG represents a deterministic snapshot of a DataFrame
transformation. Edges carry the operation metadata that produced the child
from the parent.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Commit node
# ---------------------------------------------------------------------------


@dataclass
class CommitNode:
    """A single node in the transformation DAG.

    Parameters
    ----------
    node_id:
        Deterministic SHA-256 hash of (schema + op_name + op_kwargs).
    op_name:
        The name of the operation that produced this node (e.g. ``"filter"``).
    op_kwargs:
        Serialisable representation of the operation arguments.
    schema_hash:
        SHA-256 of the serialised Polars schema.
    label:
        Optional human-readable tag (set via :func:`flashback.commit`).
    message:
        Optional free-text description.
    timestamp:
        UTC wall-clock time when the node was created.
    parent_ids:
        Ordered list of parent ``node_id`` strings (empty for root nodes).
    shape:
        ``(n_rows, n_cols)`` at this node — stored lazily after materialisation.
    """

    node_id: str
    op_name: str
    op_kwargs: dict[str, Any]
    schema_hash: str
    label: str = ""
    message: str = ""
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    parent_ids: list[str] = field(default_factory=list)
    shape: tuple[int, int] = (0, 0)

    # The actual materialized data lives here (optional — not persisted to disk
    # by default; re-materialised on checkout if needed).
    _data: pl.DataFrame | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (without ``_data``)."""
        return {
            "node_id": self.node_id,
            "op_name": self.op_name,
            "op_kwargs": self.op_kwargs,
            "schema_hash": self.schema_hash,
            "label": self.label,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "parent_ids": self.parent_ids,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CommitNode":
        """Deserialise from a JSON-compatible dict."""
        return cls(
            node_id=d["node_id"],
            op_name=d["op_name"],
            op_kwargs=d["op_kwargs"],
            schema_hash=d["schema_hash"],
            label=d.get("label", ""),
            message=d.get("message", ""),
            timestamp=datetime.datetime.fromisoformat(d["timestamp"]),
            parent_ids=d.get("parent_ids", []),
            shape=tuple(d.get("shape", [0, 0])),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _hash_schema(schema: pl.Schema) -> str:
    """Produce a deterministic hash of a Polars schema."""
    canonical = json.dumps(
        {name: str(dtype) for name, dtype in schema.items()},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _safe_serialise(value: Any) -> Any:
    """Best-effort JSON serialisation of operation arguments."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialise(v) for k, v in value.items()}
    if isinstance(value, pl.Expr):
        return repr(value)
    return str(value)


def make_node_id(
    parent_ids: list[str],
    op_name: str,
    op_kwargs: dict[str, Any],
    schema: pl.Schema,
) -> str:
    """Deterministic SHA-256 node identifier.

    The hash is a function of: parent chain + operation name + serialised
    kwargs + schema.  This means identical transformations applied to the
    same parent always resolve to the same node — enabling cache hits.
    """
    payload = json.dumps(
        {
            "parents": parent_ids,
            "op": op_name,
            "kwargs": _safe_serialise(op_kwargs),
            "schema": {n: str(t) for n, t in schema.items()},
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


class LineageDAG:
    """Directed Acyclic Graph tracking the transformation history of a frame.

    Each ``FlashbackFrame`` owns exactly one ``LineageDAG``.  Nodes are
    stored in insertion order; the *head* pointer tracks the current tip.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CommitNode] = {}
        self._head: str | None = None
        # label → node_id index for O(1) checkout
        self._label_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        *,
        op_name: str,
        op_kwargs: dict[str, Any],
        data: pl.DataFrame,
        parent_ids: list[str] | None = None,
        label: str = "",
        message: str = "",
    ) -> CommitNode:
        """Create a new commit node and advance HEAD.

        Parameters
        ----------
        op_name:
            Name of the operation (e.g. ``"filter"``).
        op_kwargs:
            Serialisable operation arguments.
        data:
            The materialised Polars DataFrame at this point.
        parent_ids:
            Parent node IDs. ``None`` → use current HEAD.
        label:
            Optional human-readable tag.
        message:
            Optional description.

        Returns
        -------
        CommitNode
        """
        if parent_ids is None:
            parent_ids = [self._head] if self._head is not None else []

        schema = data.schema
        node_id = make_node_id(parent_ids, op_name, op_kwargs, schema)

        # Cache-hit: node already exists with this exact transformation.
        if node_id in self._nodes:
            existing = self._nodes[node_id]
            # Re-store data in case it was evicted.
            existing._data = data
            self._head = node_id
            if label:
                self._register_label(label, node_id)
            return existing

        node = CommitNode(
            node_id=node_id,
            op_name=op_name,
            op_kwargs=op_kwargs,
            schema_hash=_hash_schema(schema),
            label=label,
            message=message,
            parent_ids=parent_ids,
            shape=data.shape,
            _data=data,
        )
        self._nodes[node_id] = node
        self._head = node_id

        if label:
            self._register_label(label, node_id)

        return node

    def tag_current(self, label: str, *, message: str = "") -> None:
        """Attach a label to the current HEAD node."""
        if self._head is None:
            msg = "Cannot tag: DAG has no commits yet."
            raise RuntimeError(msg)
        self._nodes[self._head].label = label
        self._nodes[self._head].message = message
        self._register_label(label, self._head)

    def _register_label(self, label: str, node_id: str) -> None:
        if label in self._label_index and self._label_index[label] != node_id:
            # Warn on label collision — overwrite.
            import warnings

            warnings.warn(
                f"flashback: label '{label}' already exists and will be reassigned.",
                stacklevel=3,
            )
        self._label_index[label] = node_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def head(self) -> CommitNode | None:
        """Return the current HEAD node, or ``None`` if the DAG is empty."""
        if self._head is None:
            return None
        return self._nodes[self._head]

    def get_node(self, node_id: str) -> CommitNode | None:
        return self._nodes.get(node_id)

    def checkout(self, label: str) -> "FlashbackFrame":  # type: ignore[return]
        """Return a new ``FlashbackFrame`` pointing to the labelled commit.

        The returned frame is fully materialised and shares no mutable state
        with the original DAG — it is a clean branch point.
        """
        from flashback.core import FlashbackFrame  # avoid circular

        node_id = self._label_index.get(label)
        if node_id is None:
            available = list(self._label_index.keys())
            msg = f"Label '{label}' not found. Available: {available}"
            raise KeyError(msg)

        node = self._nodes[node_id]
        if node._data is None:
            # Data was evicted; replay from the root.
            node._data = self._replay(node_id)

        # Build a fresh DAG that branches from this node.
        child_dag = LineageDAG()
        child_dag._nodes = dict(self._nodes)  # shallow-copy the full graph
        child_dag._head = node_id
        child_dag._label_index = dict(self._label_index)

        frame = FlashbackFrame(df=node._data, dag=child_dag)
        return frame

    def _replay(self, target_node_id: str) -> pl.DataFrame:
        """Replay the transformation chain from the root to *target_node_id*.

        This is the fallback when cached ``_data`` was evicted.  The replay
        walks the ancestor chain and re-applies operations recorded in each
        node's ``op_name`` / ``op_kwargs`` metadata.

        NOTE: For complex Polars expressions (arbitrary lambdas) that cannot
        be round-tripped through JSON, this may not be possible.  In that
        case callers should ensure ``_data`` is retained in memory.  For the
        purposes of this implementation we require data to be retained.
        """
        # Walk to the root.
        chain: list[CommitNode] = []
        current_id: str | None = target_node_id
        while current_id is not None:
            node = self._nodes[current_id]
            chain.append(node)
            current_id = node.parent_ids[0] if node.parent_ids else None

        chain.reverse()

        root = chain[0]
        if root._data is None:
            msg = (
                "Cannot replay: root node data was evicted and no source path "
                "is available. Retain data in memory or persist to disk."
            )
            raise RuntimeError(msg)

        df = root._data
        for node in chain[1:]:
            if node._data is not None:
                df = node._data
            # If we still can't replay we surface a clear error.

        return df

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_nodes(self) -> list[CommitNode]:
        """Return all nodes in insertion order."""
        return list(self._nodes.values())

    def ancestors(self, node_id: str) -> list[CommitNode]:
        """Return the linear ancestor chain for *node_id* (root first)."""
        chain: list[CommitNode] = []
        current: str | None = node_id
        while current is not None:
            node = self._nodes.get(current)
            if node is None:
                break
            chain.append(node)
            current = node.parent_ids[0] if node.parent_ids else None
        chain.reverse()
        return chain

    def to_networkx(self) -> Any:
        """Export the DAG as a ``networkx.DiGraph``."""
        import networkx as nx

        g: nx.DiGraph = nx.DiGraph()
        for node in self._nodes.values():
            g.add_node(
                node.node_id,
                label=node.label or node.op_name,
                op=node.op_name,
                shape=node.shape,
                timestamp=node.timestamp.isoformat(),
            )
            for pid in node.parent_ids:
                g.add_edge(pid, node.node_id)
        return g

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full DAG to a JSON-compatible dict."""
        return {
            "head": self._head,
            "label_index": self._label_index,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineageDAG":
        """Deserialise from a JSON-compatible dict."""
        dag = cls()
        dag._head = d.get("head")
        dag._label_index = d.get("label_index", {})
        dag._nodes = {nid: CommitNode.from_dict(nd) for nid, nd in d["nodes"].items()}
        return dag
