"""Terminal and Jupyter visualisation for flashback lineage graphs.

Renders the transformation DAG as a ``rich``-powered git-log-style tree in
the terminal, or as an SVG/HTML widget when running inside a Jupyter kernel.

Public entry point: :func:`render`.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from flashback.dag import CommitNode, LineageDAG

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOURS = {
    "root": "bright_cyan",
    "filter": "green3",
    "with_columns": "steel_blue1",
    "select": "dark_orange3",
    "join": "orchid1",
    "sort": "light_goldenrod2",
    "group_by": "light_salmon1",
    "lag": "deep_sky_blue1",
    "rolling_mean": "aquamarine1",
    "drop": "grey58",
    "rename": "thistle3",
    "load": "bright_cyan",
    "__getitem__": "grey50",
    "default": "white",
}

_OP_ICONS = {
    "root": "🌱",
    "load": "📂",
    "filter": "🔍",
    "with_columns": "➕",
    "select": "📌",
    "join": "🔗",
    "sort": "🔀",
    "group_by": "📦",
    "lag": "⏪",
    "rolling_mean": "📈",
    "drop": "🗑️",
    "rename": "✏️",
    "drop_nulls": "🧹",
    "head": "⬆️",
    "tail": "⬇️",
    "default": "⚙️",
}


def _op_colour(op: str) -> str:
    return _COLOURS.get(op, _COLOURS["default"])


def _op_icon(op: str) -> str:
    return _OP_ICONS.get(op, _OP_ICONS["default"])


def _fmt_ts(ts: datetime.datetime) -> str:
    return ts.strftime("%H:%M:%S")


def _fmt_shape(shape: tuple[int, int]) -> str:
    return f"{shape[0]:,} × {shape[1]}"


# ---------------------------------------------------------------------------
# Tree-style renderer
# ---------------------------------------------------------------------------


def _build_rich_tree(dag: "LineageDAG") -> Tree:
    """Construct a ``rich.Tree`` from the DAG."""
    head = dag.head
    if head is None:
        return Tree("[dim]empty DAG[/dim]")

    # Walk the full ancestor chain of the HEAD node.
    ancestors = dag.ancestors(head.node_id)
    if not ancestors:
        return Tree("[dim]empty DAG[/dim]")

    root_node = ancestors[0]
    label_str = (
        f" [bright_yellow]({root_node.label})[/bright_yellow]"
        if root_node.label
        else ""
    )

    root_text = Text.assemble(
        (_op_icon("load") + " ", ""),
        (root_node.op_name.upper(), _op_colour(root_node.op_name)),
        (f"  {_fmt_shape(root_node.shape)} rows", "dim"),
        (f"  [{_fmt_ts(root_node.timestamp)}]", "dim"),
        (label_str, ""),
    )
    tree = Tree(root_text)
    parent_branch = tree

    for node in ancestors[1:]:
        _add_branch(parent_branch, node, is_head=(node.node_id == head.node_id))
        # For a linear chain, each node becomes the parent of the next.
        if node.node_id == head.node_id:
            break
        # Advance to the deepest branch added.
        parent_branch = parent_branch.children[-1] if parent_branch.children else parent_branch

    return tree


def _add_branch(parent: Tree, node: "CommitNode", *, is_head: bool) -> None:
    colour = _op_colour(node.op_name)
    icon = _op_icon(node.op_name)
    label_str = (
        f" [bright_yellow]({node.label})[/bright_yellow]" if node.label else ""
    )
    head_badge = " [reverse bright_white] HEAD [/reverse bright_white]" if is_head else ""

    # Format kwargs summary (first 2 keys, truncated).
    kwargs_items = list(node.op_kwargs.items())[:2]
    kwargs_str = "  " + "  ".join(f"[dim]{k}[/dim]=[cyan]{str(v)[:24]}[/cyan]" for k, v in kwargs_items) if kwargs_items else ""

    text = Text.assemble(
        (f"{icon} ", ""),
        (node.op_name, colour),
        (kwargs_str, ""),
        (f"  {_fmt_shape(node.shape)}", "dim"),
        (f"  [{_fmt_ts(node.timestamp)}]", "dim"),
        (f"  #{node.node_id[:8]}", "dim"),
        (label_str, ""),
        (head_badge, ""),
    )
    parent.add(text)


# ---------------------------------------------------------------------------
# DAG-style renderer (compact)
# ---------------------------------------------------------------------------


def _render_dag_ascii(dag: "LineageDAG", console: Console) -> None:
    """Compact ASCII DAG view similar to ``git log --graph``."""
    head = dag.head
    if head is None:
        console.print("[dim]No commits.[/dim]")
        return

    ancestors = dag.ancestors(head.node_id)
    lines: list[str] = []
    for i, node in enumerate(ancestors):
        is_head = node.node_id == head.node_id
        connector = "●" if is_head else "○"
        pipe = "│" if i < len(ancestors) - 1 else " "
        label_tag = f" ({node.label})" if node.label else ""
        shape_str = f"{node.shape[0]:,}×{node.shape[1]}"
        lines.append(
            f"[{_op_colour(node.op_name)}]{connector}[/] {node.op_name:<18}"
            f"[dim] {shape_str:<14} #{node.node_id[:8]}[/dim]"
            f"[bright_yellow]{label_tag}[/bright_yellow]"
            f"{'  [reverse] HEAD [/reverse]' if is_head else ''}"
        )
        if i < len(ancestors) - 1:
            lines.append(f"[dim]{pipe}[/dim]")

    for line in lines:
        console.print(line)


# ---------------------------------------------------------------------------
# Jupyter / HTML renderer
# ---------------------------------------------------------------------------


def _is_jupyter() -> bool:
    """Return ``True`` if we appear to be inside a Jupyter kernel."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except ImportError:
        return False


def _render_html(dag: "LineageDAG") -> str:
    """Generate an SVG/HTML representation of the DAG for Jupyter."""
    head = dag.head
    if head is None:
        return "<p><em>No commits yet.</em></p>"

    ancestors = dag.ancestors(head.node_id)
    node_height = 64
    node_width = 280
    padding = 20
    total_height = len(ancestors) * (node_height + padding) + padding
    total_width = node_width + 80

    svg_nodes: list[str] = []
    connector_lines: list[str] = []

    for i, node in enumerate(ancestors):
        y = padding + i * (node_height + padding)
        cx = 40  # circle x
        cy = y + node_height // 2

        # Connector line to next node.
        if i < len(ancestors) - 1:
            next_cy = cy + node_height + padding
            connector_lines.append(
                f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{next_cy}" '
                f'stroke="#4a5568" stroke-width="2" stroke-dasharray="4"/>'
            )

        is_head = node.node_id == head.node_id
        circle_fill = "#38b2ac" if is_head else "#4a5568"
        text_fill = "#e2e8f0"
        label_part = f" ({node.label})" if node.label else ""
        shape_str = f"{node.shape[0]:,} × {node.shape[1]}"

        svg_nodes.append(
            f'<circle cx="{cx}" cy="{cy}" r="10" fill="{circle_fill}" />'
            f'<text x="{cx + 20}" y="{cy - 8}" font-size="13" fill="{text_fill}" '
            f'font-family="monospace" font-weight="bold">'
            f'{node.op_name}{label_part}</text>'
            f'<text x="{cx + 20}" y="{cy + 10}" font-size="11" fill="#718096" '
            f'font-family="monospace">{shape_str}  #{node.node_id[:8]}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_width}" height="{total_height}" '
        f'style="background:#1a202c;border-radius:8px;padding:8px;">'
        + "".join(connector_lines)
        + "".join(svg_nodes)
        + "</svg>"
    )
    return f'<div style="font-family:monospace">{svg}</div>'


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render(
    dag: "LineageDAG",
    *,
    style: str = "tree",
    max_width: int = 120,
) -> None:
    """Render *dag* to the terminal or Jupyter.

    Parameters
    ----------
    dag:
        The ``LineageDAG`` to visualise.
    style:
        ``"tree"`` (default) or ``"dag"`` (compact ASCII).
    max_width:
        Terminal width for rich output.
    """
    if _is_jupyter():
        from IPython.display import HTML, display  # type: ignore[import-untyped]

        display(HTML(_render_html(dag)))
        return

    console = Console(width=max_width)
    head = dag.head
    title = f"flashback lineage  •  {len(dag.list_nodes())} commits"
    if head and head.label:
        title += f"  •  HEAD → {head.label}"

    console.print()
    if style == "dag":
        console.print(Panel(title, style="bold bright_cyan"))
        _render_dag_ascii(dag, console)
    else:
        tree = _build_rich_tree(dag)
        console.print(
            Panel(
                tree,
                title=f"[bold bright_cyan]{title}[/bold bright_cyan]",
                border_style="bright_cyan",
                expand=False,
            )
        )
    console.print()
