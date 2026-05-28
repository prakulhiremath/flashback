"""Global registry for ``FlashbackFrame`` instances.

The registry holds weak references to all frames registered in the current
Python process so that :func:`flashback.checkout` and :func:`flashback.visualize`
can locate them without requiring the caller to pass an explicit frame handle.

Design notes
------------
* Frames are stored as **strong** references inside ``_frames`` — we
  intentionally keep them alive so that DAG replay from the root is always
  possible.  This is an explicit design trade-off: memory cost for
  instant checkout.  Power users can call :func:`flashback.reset` to clear.
* The registry is not thread-safe for concurrent *writes* — Python's GIL
  provides safety for most interactive / notebook workflows.  Production
  pipelines with concurrent writes should use per-frame DAGs directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flashback.core import FlashbackFrame


class _Registry:
    """Singleton-style global registry."""

    def __init__(self) -> None:
        # We store by object id to avoid duplicate registrations.
        self._frames: dict[int, "FlashbackFrame"] = {}
        # label → object id — fast reverse lookup.
        self._label_to_id: dict[str, int] = {}

    def register(self, frame: "FlashbackFrame") -> None:
        """Register or update a frame in the global store."""
        fid = id(frame)
        self._frames[fid] = frame
        # Index all labels exposed by this frame's DAG.
        from flashback.dag import LineageDAG  # avoid circular

        dag: LineageDAG = object.__getattribute__(frame, "_dag")
        for label, node_id in dag._label_index.items():
            self._label_to_id[label] = fid

    def checkout(self, label: str) -> "FlashbackFrame | None":
        """Return the frame at *label*, or ``None`` if not found."""
        fid = self._label_to_id.get(label)
        if fid is None:
            return None
        frame = self._frames.get(fid)
        if frame is None:
            return None
        dag = object.__getattribute__(frame, "_dag")
        try:
            return dag.checkout(label)
        except KeyError:
            return None

    def list_labels(self) -> list[str]:
        """Return all known checkpoint labels across all registered frames."""
        return sorted(self._label_to_id.keys())

    def all_frames(self) -> list["FlashbackFrame"]:
        """Return all currently registered frames (in registration order)."""
        return list(self._frames.values())

    def clear(self) -> None:
        """Remove all registered frames and labels."""
        self._frames.clear()
        self._label_to_id.clear()


#: Module-level singleton — import this in other modules.
_global_registry = _Registry()
