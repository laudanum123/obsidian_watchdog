from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

class FsEvent(BaseModel):
    path: Path
    kind: str  # "create" | "modify" | "delete"
    ts: datetime = Field(default_factory=datetime.utcnow)

class Patch(BaseModel):
    before: str
    after: str
    comment: Optional[str] = None  # free‑form
    agent: Optional[str] = None  # who produced it 

class EditScriptAction(BaseModel):
    script: str = Field(default="", description="The new version of the text block after changes.")
    original_context_for_script: str = Field(default="", description="The exact, verbatim lines from the Original Note that the 'script' field is intended to replace.")
    comment: str = Field(description="Explanation of why the change was made or not made.")
    agent_name: str = Field(description="Name of the agent proposing the edit.")
    no_change_needed: bool = Field(default=False, description="Set to true if no edit script is generated and no change is intended.")

# New Models based on config.yaml structure

class DatabaseConfig(BaseModel):
    name: str

class KVStoreConfig(BaseModel):
    name: str

class ApplicationSettings(BaseModel):
    default_llm_model: Optional[str] = None
    # Add other agent-specific settings here as needed, e.g.:
    # daily_digest_agent: Optional[Dict[str, Any]] = None 

class EmbeddingModelConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str
    api_key: str
    dimensions: Optional[int] = None

class DBPopulationConfig(BaseModel):
    ignore_patterns: List[str] = Field(default_factory=list)
    embedding_batch_size: int = 10
    force_reembed_all: bool = False

class VaultConfig(BaseModel):
    vault_path: Optional[Path] = None # Optional: can be overridden by env var
    database: DatabaseConfig
    kv_store: KVStoreConfig
    application_settings: ApplicationSettings
    embedding_model: EmbeddingModelConfig
    db_population: DBPopulationConfig 