# ⚡ flashback

> **Git for Datasets** — time-travel debugging and transformation lineage tracking for pandas & Polars.

[![CI](https://github.com/flashback-dev/flashback/actions/workflows/ci.yml/badge.svg)](https://github.com/flashback-dev/flashback/actions)
[![PyPI](https://img.shields.io/pypi/v/flashback.svg)](https://pypi.org/project/flashback)
[![Python](https://img.shields.io/pypi/pyversions/flashback.svg)](https://pypi.org/project/flashback)
[![Coverage](https://codecov.io/gh/flashback-dev/flashback/branch/main/graph/badge.svg)](https://codecov.io/gh/flashback-dev/flashback)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```
📂 load  ──▶  🔍 filter  ──▶  ➕ with_columns  ──▶  ⏪ lag  ──▶  HEAD
                  │
              (before-lag)  ◀── fb.checkout("before-lag")
```

---

## Why this exists

Every ML researcher has asked: **"Why did my metric change?"** Nobody knows.

You ran a 6-hour training job, the Sharpe ratio dropped from 1.4 to 0.9, and
somewhere between the raw tick data and the feature matrix a silent
transformation introduced look-ahead bias. You have no idea where.

**DVC is too heavy** — it versions entire files with S3 backends, CI pipelines,
and YAML configs.  You don't want to learn a new orchestration system; you
want to know what happened to column `price_lag1` between step 3 and step 7.

**Git doesn't understand columns.** `git diff` on a Parquet file is binary
noise.  It cannot tell you "this `.filter()` removed 412 rows" or "this
`.with_columns()` introduced a null in 3% of rows."

**flashback fixes this.**

It wraps your DataFrame in a zero-cost proxy that records every transformation
as a node in an in-memory Directed Acyclic Graph (DAG).  Each node is
identified by a deterministic SHA-256 hash of the schema + operation
arguments, giving you:

- **Instant time-travel** — `fb.checkout("before-lag")` returns the exact
  frame at that checkpoint with no I/O unless you ask for it.
- **Structural diffing** — `frame.diff(other)` shows you exactly which rows
  were added or removed between any two checkpoints.
- **Beautiful lineage views** — `fb.visualize()` renders a `rich`-powered
  git-log-style tree in your terminal, or an SVG graph in Jupyter.
- **Reproducibility** — identical transformations applied to identical data
  always produce the same node ID — transformations are deterministic by
  construction.

---

## Install

```bash
pip install flashback
# or, if you use uv (recommended):
uv add flashback
```

**Requirements:** Python ≥ 3.10, Polars ≥ 0.20, pandas ≥ 2.0.

---

## Quickstart

```python
import flashback as fb

# ── 1. Load any source ──────────────────────────────────────────────────────
df = fb.load("trades.parquet")          # Parquet
df = fb.load("prices.csv")             # CSV
df = fb.load(my_polars_df)             # existing Polars DataFrame
df = fb.load(my_pandas_df)             # existing Pandas DataFrame

# ── 2. Transform — every step is recorded automatically ─────────────────────
df = df.filter(fb.col("price") > 0)
df = df.with_columns(
    (fb.col("price") * fb.col("volume")).alias("notional")
)

# Tag a checkpoint before the next risky operation.
df = df.tag("before-lag")

df = df.lag("price", 1)               # sugar for shift(-1) + tracking
df = df.rolling_mean("notional", 5)

# ── 3. Time-travel ──────────────────────────────────────────────────────────
df_clean = fb.checkout("before-lag")  # ← instant; no disk I/O

# ── 4. See what broke your Sharpe ratio ─────────────────────────────────────
fb.visualize()
```

Terminal output:

```
╭─ flashback lineage  •  4 commits  •  HEAD → rolling_mean ──────────────────╮
│                                                                             │
│  📂 LOAD  5,000 rows × 4 cols  [14:03:01]                                  │
│  │                                                                          │
│  ├─ 🔍 filter  arg_0=...col("price")...  4,823 rows × 4 cols  #a1b2c3d4   │
│  │                                                                          │
│  ├─ ➕ with_columns  arg_0=...alias("notional")  4,823 rows × 5  #e5f6a7  │
│  │                                                                          │
│  ├─ ⏪ lag  column='price'  n=1  4,823 rows × 6  [before-lag]  #b8c9d0    │
│  │                                                                          │
│  └─ 📈 rolling_mean  window=5  4,823 rows × 7 ● HEAD  #01e2f3a4           │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

## API Reference

### `fb.load(source, *, label=None, track=True)`

Load a DataFrame from a file path, Polars DataFrame, or Pandas DataFrame and
begin tracking its lineage.

| Param | Type | Description |
|-------|------|-------------|
| `source` | `str \| pl.DataFrame \| pd.DataFrame \| FlashbackFrame` | Data source |
| `label` | `str \| None` | Human-readable root label (default: filename stem or `"root"`) |
| `track` | `bool` | Register with the global registry (default: `True`) |

**Supported formats:** `.parquet`, `.csv`, `.json`, `.ndjson`, `.ipc`, `.arrow`

---

### `fb.col(name)`

Alias for `polars.col`.  Use inside transform chains for IDE-friendly imports:

```python
df = df.filter(fb.col("price") > 0)
```

---

### `fb.commit(frame, label, *, message="")`

Tag the current state of `frame` with a human-readable label — analogous to
`git tag`.

```python
df = fb.commit(df, "before-normalise", message="Raw features, no scaling")
```

Or use the method form:

```python
df = df.tag("before-normalise", message="Raw features, no scaling")
```

---

### `fb.checkout(label, *, frame=None)`

Time-travel to a named checkpoint.  Returns a new `FlashbackFrame` at that
exact state, fully materialised.

```python
df_original = fb.checkout("before-normalise")
```

If `frame` is provided, searches only that frame's lineage.  Otherwise,
searches the global registry.

---

### `fb.visualize(frame=None, *, style="tree", max_width=120)`

Render the transformation lineage.

- `style="tree"` — rich tree with icons, timestamps, shapes, node IDs.
- `style="dag"` — compact ASCII graph (`git log --graph` style).
- In Jupyter, automatically falls back to an SVG/HTML widget.

---

### `FlashbackFrame.lag(column, n=1, *, alias=None)`

Shift `column` by `n` periods with a tracked checkpoint.

```python
df = df.lag("price", 1)                    # → price_lag1
df = df.lag("price", 3, alias="price_t3")  # → price_t3
```

---

### `FlashbackFrame.rolling_mean(column, window, *, alias=None, min_periods=None)`

Rolling mean over `window` periods with lineage tracking.

```python
df = df.rolling_mean("notional", 20)  # → notional_rmean20
```

---

### `FlashbackFrame.diff(other)`

Structural diff between two frames.  Returns a Polars DataFrame with a `_diff`
column of `"added"` / `"removed"`.

```python
delta = df_now.diff(df_old)
print(delta.filter(pl.col("_diff") == "removed"))
```

---

### `FlashbackFrame.history()`

Return the full transformation chain as a list of dicts (root → HEAD):

```python
for step in df.history():
    print(step["op_name"], step["shape"], step["label"])
```

---

## Persistence

Lineage graphs can be saved to and loaded from disk:

```python
from flashback.storage import Storage

store = Storage(".flashback")  # or Storage.from_cwd()
store.save(df, frame_id="experiment-001")

# Later, in another session:
df = store.load("experiment-001")
```

The `.flashback/` directory layout:

```
.flashback/
├── config.json
├── graphs/
│   └── experiment-001.json   # serialised DAG
└── cache/
    └── <node_id>.parquet     # materialised node snapshots
```

---

## How it works

```
┌──────────────────────────────────────────────────────────┐
│  FlashbackFrame                                          │
│                                                          │
│  ┌──────────────┐    intercept    ┌───────────────────┐  │
│  │  Polars API  │ ─────────────▶ │   LineageDAG      │  │
│  │  .filter()   │                │                   │  │
│  │  .sort()     │  record node   │  root ──▶ filter  │  │
│  │  .join()     │ ◀──────────── │         ──▶ sort  │  │
│  └──────────────┘                │         ──▶ join  │  │
│         │                        └───────────────────┘  │
│         ▼                                               │
│  polars.DataFrame  (unchanged; Polars still optimises)  │
└──────────────────────────────────────────────────────────┘
```

**Node identity** is a 20-character hex SHA-256 of:
```json
{
  "parents": ["<parent_node_id>"],
  "op": "filter",
  "kwargs": {"arg_0": "[(col(\"price\")) > (0)]"},
  "schema": {"id": "Int64", "price": "Float64", ...}
}
```

This means:
- Identical pipelines on identical data always hash to the same node → instant
  cache hits.
- Changing *any* argument or parent state produces a *different* hash → no
  silent collisions.

---

## Development

```bash
git clone https://github.com/flashback-dev/flashback
cd flashback
pip install -e ".[dev]"

# Lint
ruff check flashback tests
ruff format --check flashback tests

# Type-check
mypy flashback

# Test with coverage
pytest
```

The CI matrix runs across **Ubuntu × macOS × Windows** and **Python 3.10 –
3.13** with a hard 90% coverage threshold.

---

## Roadmap

- [ ] **Branching** — `fb.branch("experiment-A")` for parallel pipeline exploration
- [ ] **Merge** — reconcile two branches at the DAG level
- [ ] **Remote storage** — push/pull lineage graphs to S3 / GCS
- [ ] **Streaming Polars** — track lazy plans before `.collect()`
- [ ] **Notebook integration** — `%load_ext flashback` magic with live DAG sidebar
- [ ] **Export to DVC** — generate `.dvc` stage files from a flashback DAG

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Built with <a href="https://pola.rs">Polars</a> · <a href="https://github.com/Textualize/rich">Rich</a> · <a href="https://networkx.org">NetworkX</a>
</p>
