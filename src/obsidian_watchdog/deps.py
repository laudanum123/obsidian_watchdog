from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
# For DuckDB and TinyDB, you'll need to install them: 
# uv pip install duckdb tinydb
import duckdb
from tinydb import TinyDB
import time

RECENTLY_MODIFIED_TTL_SECONDS = 5 # Ignore events for files modified by agent within this window

@dataclass
class VaultCtx:
    root: Path
    db: duckdb.DuckDBPyConnection       # embeddings & metadata
    kv: TinyDB                           # simple KV store (counters, flags)
    config: dict[str, Any]               # YAML‑loaded settings 
    embedding_dimensions: int            # Dimension of embeddings (e.g., 1024)
    # Stores (path_str, modification_timestamp) to ignore self-generated events
    _agent_modified_paths: Dict[str, float] = field(default_factory=dict)

    def add_agent_modified_path(self, path_str: str):
        """Records that an agent has modified a path."""
        self._agent_modified_paths[path_str] = time.monotonic()
        # print(f"[VaultCtx] Added to agent_modified_paths: {path_str}") # For debugging

    def was_recently_modified_by_agent(self, path_str: str) -> bool:
        """Checks if a path was recently modified by an agent."""
        # Clean up old entries first (optional, but good practice)
        current_time = time.monotonic()
        for p, ts in list(self._agent_modified_paths.items()): # Iterate over a copy
            if current_time - ts > RECENTLY_MODIFIED_TTL_SECONDS:
                # print(f"[VaultCtx] Removing expired from agent_modified_paths: {p}") # For debugging
                del self._agent_modified_paths[p]
        
        if path_str in self._agent_modified_paths:
            # print(f"[VaultCtx] Path {path_str} was recently modified by agent.") # For debugging
            return True
        return False 