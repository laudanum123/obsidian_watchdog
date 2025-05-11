import asyncio
import hashlib
import os
from datetime import datetime, timezone  # Added timezone
from pathlib import Path


import logfire
import yaml  # For loading config, install with: uv pip install pyyaml
from openai import AsyncOpenAI

from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.runner import run_observer

logfire.configure()


# Assuming your project structure is:
# project_root/
#  ├── main.py
#  ├── runner.py
#  ├── deps.py
#  ├── models.py
#  ├── tools_common.py
#  ├── router.py
#  ├── config.yaml
#  └── agents/
#      ├── __init__.py
#      └── backlinker.py

# Use the official OpenAI library for creating embeddings




# Attempt to import database drivers
try:
    import duckdb
except ImportError:
    print("Warning: duckdb is not installed. `uv pip install duckdb`")
    duckdb = None

try:
    from tinydb import Query, TinyDB
    from tinydb.middlewares import CachingMiddleware
    from tinydb.storages import JSONStorage
except ImportError:
    print("Warning: tinydb is not installed. `uv pip install tinydb`")
    TinyDB = None
    Query = None

DEFAULT_VAULT_PATH = "C:/Users/Joerg/My Documents/obsidian/Private/" # Default, can be overridden by config or env var
DEFAULT_CONFIG_PATH = "config.yaml"
DB_SCHEMA_VERSION = 5 # Increment if schema changes (e.g. storing text content)


def load_configuration(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration from {config_path}: {e}")
        return {}

def get_embedding_client(config: dict) -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for embeddings, configured for LM Studio."""
    embedding_config = config.get("embedding_model", {})
    # model_name here is for informational logging; the actual model is specified in embeddings.create()
    model_name_info = embedding_config.get("model_name", "default-embedding-model")
    base_url = embedding_config.get("base_url", "http://localhost:1234/v1") # Default LMStudio URL
    api_key = embedding_config.get("api_key", "not-needed")

    print(f"[Embedding] Initializing AsyncOpenAI client for embeddings (model info: {model_name_info}) at {base_url}")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

async def populate_notes_db(db: duckdb.DuckDBPyConnection, vault_root: Path, embed_client: AsyncOpenAI, config: dict): # type: ignore
    """Scans the vault, generates embeddings, and populates the DuckDB notes table."""
    print("[DB Population] Starting to populate/update notes database...")

    # --- BEGIN MODIFICATION: Define embedding_dimensions early ---
    embedding_model_config = config.get("embedding_model", {})
    embedding_api_model_name = embedding_model_config.get("model_name", "text-embedding-mxbai-embed-large-v1") # Model for API call
    embedding_dimensions = embedding_model_config.get("dimensions") # Optional dimensions

    if embedding_dimensions is None:
        # Default to 1024 if not specified, as it's common for mxbai-embed-large-v1
        # It's CRUCIAL that this matches the actual output dimension of your embedding model.
        embedding_dimensions = 1024 
        print(f"[DB Population] 'dimensions' not found in embedding_model config, defaulting to {embedding_dimensions}. Verify this matches your model's output.")
    else:
        print(f"[DB Population] Using 'dimensions': {embedding_dimensions} from config for embedding schema and HNSW index.")
    # --- END MODIFICATION ---

    db.execute("CREATE TABLE IF NOT EXISTS db_version (version INTEGER PRIMARY KEY)")
    current_db_version_row = db.execute("SELECT version FROM db_version ORDER BY version DESC LIMIT 1").fetchone()
    current_db_version = current_db_version_row[0] if current_db_version_row else 0

    if current_db_version < DB_SCHEMA_VERSION:
        print(f"[DB Population] DB schema version mismatch (DB: {current_db_version}, Code: {DB_SCHEMA_VERSION}). Re-creating notes table.")
        db.execute("DROP TABLE IF EXISTS notes")
        # Schema: path, content_hash, modified_at (file), last_embedded_at (when embedding was generated), embedding (FLOAT vector)
        # Added raw_content for potential future use by agents without re-reading file, though it increases DB size.
        # embedding_model_config = config.get("embedding_model", {}) # No longer needed here
        # embedding_api_model_name = embedding_model_config.get("model_name", "text-embedding-mxbai-embed-large-v1") # No longer needed here
        # embedding_dimensions = embedding_model_config.get("dimensions") # No longer needed here

        # if embedding_dimensions is None: # Logic moved up
        #     embedding_dimensions = 1024 
        #     print(f"[DB Population] 'dimensions' not found in embedding_model config, defaulting to {embedding_dimensions}. Verify this matches your model's output.")
        # else:
        #     print(f"[DB Population] Using 'dimensions': {embedding_dimensions} from config for embedding schema and HNSW index.")

        create_table_sql = f"""
        CREATE TABLE notes (
            path VARCHAR PRIMARY KEY,
            content_hash VARCHAR NOT NULL,
            modified_at TIMESTAMP NOT NULL,
            last_embedded_at TIMESTAMP NOT NULL,
            raw_content TEXT, 
            embedding FLOAT[{embedding_dimensions}]
        );
        """
        db.execute(create_table_sql)
        db.execute("INSERT OR REPLACE INTO db_version (version) VALUES (?)", [DB_SCHEMA_VERSION])
        print("[DB Population] 'notes' table created/updated with new schema.")
        existing_notes_data = {} # No existing data if table was just created
    else:
        existing_notes_data = {}
        try:
            results = db.execute("SELECT path, content_hash, last_embedded_at FROM notes").fetchall()
            for row in results:
                # Ensure timestamp is timezone-aware for proper comparison
                embedded_at_ts = row[2]
                if isinstance(embedded_at_ts, str):
                    embedded_at_ts = datetime.fromisoformat(embedded_at_ts)
                if embedded_at_ts and embedded_at_ts.tzinfo is None:
                    embedded_at_ts = embedded_at_ts.replace(tzinfo=timezone.utc) # Assume UTC if not specified
                existing_notes_data[row[0]] = {"hash": row[1], "embedded_at": embedded_at_ts}
            print(f"[DB Population] Found {len(existing_notes_data)} existing note records in DB for comparison.")
        except duckdb.CatalogException:
            print("[DB Population] 'notes' table not found (unexpected after schema check). Will proceed as if empty.")
    
    notes_to_process = []
    population_config = config.get("db_population", {})
    # embedding_model_config = config.get("embedding_model", {}) # No longer needed here, defined above
    # embedding_api_model_name = embedding_model_config.get("model_name", "text-embedding-mxbai-embed-large-v1") # No longer needed here, defined above
    # embedding_dimensions = embedding_model_config.get("dimensions") # No longer needed here, defined above

    ignore_patterns = population_config.get("ignore_patterns", [".obsidian/", ".git/", "ai_logs/", "node_modules/"])
    process_all_force = population_config.get("force_reembed_all", False)
    if process_all_force:
        print("[DB Population] force_reembed_all is TRUE. All notes will be re-embedded.")

    print(f"[DB Population] Scanning vault: {vault_root} (ignoring: {ignore_patterns}) for markdown files...")
    for filepath in vault_root.glob("**/*.md"):
        try:
            relative_path_str = str(filepath.relative_to(vault_root))
            if any(relative_path_str.startswith(pattern) for pattern in ignore_patterns):
                continue

            m_time_naive = datetime.fromtimestamp(filepath.stat().st_mtime)
            m_time_utc = m_time_naive.replace(tzinfo=timezone.utc) # Make it timezone-aware (UTC)
            
            content = filepath.read_text(encoding="utf-8")
            current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            existing_data = existing_notes_data.get(relative_path_str)
            should_process = False
            if process_all_force:
                should_process = True
            elif not existing_data:
                should_process = True
                print(f"[DB Population] New note found: {relative_path_str}")
            elif existing_data["hash"] != current_hash:
                should_process = True
                print(f"[DB Population] Content changed (hash mismatch) for: {relative_path_str}")
            elif existing_data["embedded_at"] and m_time_utc > existing_data["embedded_at"]:
                should_process = True
                print(f"[DB Population] Note modified after last embedding for: {relative_path_str}")
            
            if should_process:
                notes_to_process.append({
                    "path": relative_path_str,
                    "content": content,
                    "modified_at": m_time_utc,
                    "hash": current_hash
                })
        except Exception as e:
            print(f"[DB Population] Error processing file {filepath} during scan: {e}")

    if not notes_to_process:
        print("[DB Population] No new or modified notes to process for embedding.")
    else:
        print(f"[DB Population] Found {len(notes_to_process)} notes to generate/update embeddings for.")
        batch_size = population_config.get("embedding_batch_size", 10)
        for i in range(0, len(notes_to_process), batch_size):
            batch = notes_to_process[i:i + batch_size]
            contents_to_embed = [note["content"] for note in batch]
            print(f"[DB Population] Generating embeddings for batch {i // batch_size + 1} of {(len(notes_to_process) + batch_size - 1) // batch_size} (size: {len(batch)}). Model: {embedding_api_model_name}...")
            try:
                embedding_params = {"input": contents_to_embed, "model": embedding_api_model_name}
                if embedding_dimensions: # Add dimensions if specified in config
                    embedding_params["dimensions"] = embedding_dimensions
                
                raw_embeddings_response = await embed_client.embeddings.create(**embedding_params)
                embeddings_vectors = [item.embedding for item in raw_embeddings_response.data]
                
                db_batch_data = []
                for note, emb_vector in zip(batch, embeddings_vectors):
                    db_batch_data.append((
                        note["path"],
                        note["hash"],
                        note["modified_at"],
                        datetime.now(timezone.utc),  # last_embedded_at (current time)
                        note["content"], # Storing raw_content
                        emb_vector
                    ))
                # UPSERT logic for DuckDB
                db.executemany("INSERT INTO notes (path, content_hash, modified_at, last_embedded_at, raw_content, embedding) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET content_hash=excluded.content_hash, modified_at=excluded.modified_at, last_embedded_at=excluded.last_embedded_at, raw_content=excluded.raw_content, embedding=excluded.embedding", db_batch_data)
                print(f"[DB Population] Successfully inserted/updated batch {i // batch_size + 1} into DB.")
            except Exception as e:
                print(f"[DB Population] Error generating embeddings or inserting batch {i // batch_size + 1}: {e}")

    # Prune notes from DB that no longer exist in the vault
    # This part needs to run regardless of whether notes_to_process was empty
    print("[DB Population] Starting pruning check for deleted notes...")
    all_disk_paths_set = {str(fp.relative_to(vault_root)) for fp in vault_root.glob("**/*.md") if not any(str(fp.relative_to(vault_root)).startswith(p) for p in ignore_patterns)}
    db_paths_to_prune = [path for path in existing_notes_data if path not in all_disk_paths_set]
    if db_paths_to_prune:
        print(f"[DB Population] Pruning {len(db_paths_to_prune)} notes from DB that no longer exist on disk...")
        # DuckDB can use a list of tuples for DELETE ... WHERE path IN (SELECT column1 FROM (VALUES (?), (?), ...))
        # Or simpler, one by one for clarity if performance isn't critical for pruning few items
        for path_to_prune in db_paths_to_prune:
            db.execute("DELETE FROM notes WHERE path = ?", [path_to_prune])
        print("[DB Population] Pruning complete.")
    else:
        print("[DB Population] No notes found to prune from the database.")
    
    # After all data modifications (inserts, updates, deletes), create/update HNSW index for embeddings
    # This is crucial for performant similarity searches.
    print("[DB Population] Attempting to create/update HNSW index for note embeddings...")
    try:
        # For file-based databases, experimental persistence must be enabled for HNSW indexes.
        # This setting is connection-specific.
        db.execute("SET hnsw_enable_experimental_persistence = true;")

        # --- BEGIN DIAGNOSTIC --- 
        print("[DB Population DIAGNOSTIC] Checking 'notes' table structure before creating HNSW index...")
        try:
            table_info = db.execute("SELECT column_name, data_type FROM duckdb_columns() WHERE table_name = 'notes' AND column_name = 'embedding'").fetchone()
            if table_info:
                print(f"[DB Population DIAGNOSTIC] 'embedding' column type: {table_info[1]}")
                if table_info[1] != f'FLOAT[{embedding_dimensions}]':
                    print(f"[DB Population DIAGNOSTIC] MISMATCH! Expected FLOAT[{embedding_dimensions}], got {table_info[1]}")
            else:
                print("[DB Population DIAGNOSTIC] 'embedding' column not found in 'notes' table details.")
        except Exception as diag_e:
            print(f"[DB Population DIAGNOSTIC] Error fetching table structure: {diag_e}")
        # --- END DIAGNOSTIC --- 
        
        # Create HNSW index for cosine similarity search on the 'embedding' column.
        # 'metric = cosine' is used because array_cosine_similarity orders by 1 - cosine_distance.
        # The index helps find smallest cosine_distance, which is largest array_cosine_similarity.
        # IF NOT EXISTS is used to avoid errors if the index already exists (e.g., from a previous run).
        # --- BEGIN MODIFICATION: Use f-string for dimension in CREATE INDEX ---
        db.execute("""
        CREATE INDEX IF NOT EXISTS hnsw_embedding_idx 
        ON notes USING HNSW (embedding) 
        WITH (metric = 'cosine', M = 16, ef_construction = 100);
        """) # Note: HNSW in DuckDB implicitly uses the column's defined dimension (e.g., FLOAT[1024])
        # The explicit FLOAT[N] syntax is primarily for the CREATE TABLE statement.
        # The HNSW indexer will infer the dimension from the table's schema.
        # No need to specify [N] again in `USING HNSW (embedding)`. We trust it infers from `embedding FLOAT[1024]`.
        # However, the error suggests it's not, so let's ensure the variable is in scope and correct.
        # The key is that `embedding_dimensions` used in `CREATE TABLE` must be correct.
        # The error might be a bit misleading if the actual column type isn't FLOAT[N] as expected.
        # --- END MODIFICATION ---

        # Default M=16, ef_construction=100. You can tune these.
        # ef_search can be set at query time: SET hnsw_ef_search = X;
        print("[DB Population] HNSW index 'hnsw_embedding_idx' on 'notes.embedding' ensured.")
    except Exception as e:
        print(f"[DB Population] Warning: Could not create/update HNSW index: {e}")
        if "FLOAT[N]" in str(e):
            print("[DB Population] Hint: This error often means the 'embedding_dimensions' in your config.yaml (or the default 1024 if not set) does not match the actual output dimension of your embedding model, or the table schema needs to be updated to FLOAT[<dimension>].")
        print("[DB Population] Hint: Ensure the VSS extension is correctly loaded and DuckDB version supports these features.")

    # --- BEGIN DIAGNOSTIC FOR MISSING EMBEDDING --- 
    try:
        print("[DB Population DIAGNOSTIC] Checking for 'Data Insights Findings.md' post-population...")
        specific_note_check = db.execute("SELECT path, embedding IS NOT NULL AS has_embedding, array_length(embedding) as embedding_len FROM notes WHERE path = 'Data Insights Findings.md'").fetchone()
        if specific_note_check:
            print(f"[DB Population DIAGNOSTIC] Found 'Data Insights Findings.md': Path='{specific_note_check[0]}', HasEmbedding='{specific_note_check[1]}', EmbeddingLength='{specific_note_check[2]}'")
        else:
            print("[DB Population DIAGNOSTIC] 'Data Insights Findings.md' NOT FOUND in notes table after population.")
        
        total_notes_count = db.execute("SELECT COUNT(*) FROM notes").fetchone()
        print(f"[DB Population DIAGNOSTIC] Total notes in DB after population: {total_notes_count[0] if total_notes_count else 'Error'}")
        
        sample_paths = db.execute("SELECT path FROM notes LIMIT 5").fetchall()
        print(f"[DB Population DIAGNOSTIC] Sample paths in DB: {sample_paths}")

    except Exception as diag_e_post:
        print(f"[DB Population DIAGNOSTIC] Error during post-population check: {diag_e_post}")
    # --- END DIAGNOSTIC FOR MISSING EMBEDDING --- 

    print("[DB Population] Database population/update process finished.")

async def initialize_vault_context(config: dict) -> VaultCtx:
    """Initializes the VaultContext based on configuration."""
    
    # Determine Vault Path
    # Priority: 1. config.yaml, 2. OBSIDIAN_VAULT_PATH env var, 3. DEFAULT_VAULT_PATH
    vault_path_str = config.get("vault_path", os.environ.get("OBSIDIAN_VAULT_PATH", DEFAULT_VAULT_PATH))
    vault_root = Path(vault_path_str).expanduser().resolve()
    
    print(f"Target Obsidian Vault: {vault_root}")
    if not vault_root.is_dir():
        print(f"Error: Vault path {vault_root} is not a directory or does not exist. Please check your configuration.")
        # Attempt to create it for robustness, or exit
        try:
            print(f"Attempting to create vault directory: {vault_root}")
            vault_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Could not create vault directory {vault_root}: {e}")
            exit(1) # Or raise an exception

    # Initialize DuckDB connection
    db_conn = None
    if duckdb:
        db_path = vault_root / config.get("database", {}).get("name", "obsidian_embeddings.db")
        try:
            print(f"Initializing DuckDB at: {db_path}")
            db_conn = duckdb.connect(database=str(db_path), read_only=False)
            
            # Attempt to install and load the vss extension
            try:
                print("[DB Setup] Attempting to install and load 'vss' extension for DuckDB...")
                db_conn.execute("INSTALL 'vss';")
                db_conn.execute("LOAD 'vss';")
                print("[DB Setup] 'vss' extension loaded successfully (or already installed).")
            except Exception as e:
                print(f"[DB Setup] Warning: Could not install/load 'vss' extension for DuckDB: {e}. Similarity search might be slower or unavailable.")

            # Embedding client setup
            embed_client_config = config.get("embedding_model", {})
            if not embed_client_config.get("provider") or not embed_client_config.get("model_name"):
                print("[Warning] Embedding model provider or model_name not configured in config.yaml. DB population will be skipped.")
            else:
                embed_client = get_embedding_client(config)
                # Run populate_notes_db synchronously for startup; consider making main async if this is too slow.
                # For now, we'll create a new event loop for this specific async task.
                try:
                    await populate_notes_db(db_conn, vault_root, embed_client, config)
                except RuntimeError as e:
                    if "cannot be called when another loop is running" in str(e):
                        # This specific error should no longer occur with the new structure
                        print(f"[DB Population] Error: {e}. This was unexpected with the async refactor.")
                    else: 
                        raise
            print(f"DuckDB connection established and population attempted: {db_path}")
        except Exception as e:
            print(f"Error initializing DuckDB or populating notes: {e}")
            if db_conn: 
                db_conn.close() # Close if opened before error
            db_conn = None
    else:
        print("DuckDB not available. Some features requiring a database will be disabled.")

    # Initialize TinyDB for Key-Value store (e.g., for logging patches)
    kv_store = None
    if TinyDB:
        kv_store_path = vault_root / config.get("kv_store", {}).get("name", "agent_kv_log.json")
        try:
            print(f"Initializing TinyDB at: {kv_store_path}")
            # Using CachingMiddleware for potentially better performance with frequent writes/reads
            kv_store = TinyDB(kv_store_path, storage=CachingMiddleware(JSONStorage))
            print(f"TinyDB connection established: {kv_store_path}")
        except Exception as e: # Catch generic TinyDB errors
            print(f"Error initializing TinyDB at {kv_store_path}: {e}")
            kv_store = None
    else:
        print("TinyDB not available. Some features requiring a KV store will be disabled.")
        
    app_config = config.get("application_settings", {})

    # --- BEGIN MODIFICATION: Determine embedding_dimensions for VaultCtx ---
    # This logic is similar to populate_notes_db, ensuring consistency.
    embedding_model_config_for_ctx = config.get("embedding_model", {})
    final_embedding_dimensions = embedding_model_config_for_ctx.get("dimensions")
    if final_embedding_dimensions is None:
        final_embedding_dimensions = 1024 # Default, should match populate_notes_db logic
        print(f"[VaultCtx Init] 'dimensions' not found in embedding_model config for VaultCtx, defaulting to {final_embedding_dimensions}.")
    else:
        print(f"[VaultCtx Init] Using 'dimensions': {final_embedding_dimensions} for VaultCtx.")
    # --- END MODIFICATION ---

    return VaultCtx(
        root=vault_root,
        db=db_conn,
        kv=kv_store,
        config=app_config, # Pass through application-specific settings from config
        embedding_dimensions=final_embedding_dimensions
    )

async def main_async_runner(): # Renamed from main to avoid conflict with populate_notes_db call if main becomes async
    """Main entry point for the Obsidian Agent runner."""
    print("Starting Obsidian Agent...")
    config_file_path = os.environ.get("AGENT_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    print(f"Loading configuration from: {config_file_path}")
    config = load_configuration(config_file_path)

    # Initialize VaultCtx (which now includes DB population)
    vault_context = await initialize_vault_context(config)

    if not vault_context.db:
        print("CRITICAL: DuckDB context not available. Some agent functionalities might fail. Exiting watcher.")
        return # Don't start watcher if DB is essential and failed to init

    try:
        await run_observer(str(vault_context.root), vault_context)
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
    finally:
        print("Obsidian Agent is shutting down.")
        if vault_context.db:
            try:
                vault_context.db.close()
                print("DuckDB connection closed.")
            except Exception as e: # Catch if already closed or other errors
                print(f"Error closing DuckDB connection: {e}")
        if vault_context.kv:
            try:
                vault_context.kv.close()
                print("TinyDB connection closed.")
            except Exception as e:
                print(f"Error closing TinyDB connection: {e}")


if __name__ == "__main__":
    # For Pydantic V2 compatibility with datetimes in Pydantic AI, if encountering issues
    # from pydantic import ConfigDict
    # from pydantic_ai.models import _Model as PydanticAIModel
    # PydanticAIModel.model_config = ConfigDict(arbitrary_types_allowed=True)
    
    asyncio.run(main_async_runner()) 