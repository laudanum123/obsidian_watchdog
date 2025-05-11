from dataclasses import dataclass
from pathlib import Path
from typing import Any
# For DuckDB and TinyDB, you'll need to install them: 
# uv pip install duckdb tinydb
import duckdb
from tinydb import TinyDB

@dataclass
class VaultCtx:
    root: Path
    db: duckdb.DuckDBPyConnection       # embeddings & metadata
    kv: TinyDB                           # simple KV store (counters, flags)
    config: dict[str, Any]               # YAML‑loaded settings 
    embedding_dimensions: int            # Dimension of embeddings (e.g., 1024) 