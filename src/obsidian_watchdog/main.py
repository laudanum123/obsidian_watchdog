import asyncio
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
import argparse

import logfire
import yaml
from openai import AsyncOpenAI

from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.runner import run_observer
from obsidian_watchdog.notes_db_manager import populate_notes_db, _ensure_db_schema
from obsidian_watchdog.agents.backlinker import run_backlinker_for_all_notes

logfire.configure()

DEFAULT_VAULT_PATH = "C:/Users/Joerg/My Documents/obsidian/Private/"
DEFAULT_CONFIG_PATH = "config.yaml"

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
    model_name_info = embedding_config.get("model_name", "default-embedding-model")
    base_url = embedding_config.get("base_url", "http://localhost:1234/v1")
    api_key = embedding_config.get("api_key", "not-needed")

    print(f"[Embedding] Initializing AsyncOpenAI client for embeddings (model info: {model_name_info}) at {base_url}")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

async def initialize_vault_context(config: dict) -> VaultCtx:
    """Initializes the VaultContext based on configuration."""
    
    vault_path_str = config.get("vault_path", os.environ.get("OBSIDIAN_VAULT_PATH", DEFAULT_VAULT_PATH))
    vault_root = Path(vault_path_str).expanduser().resolve()
    
    print(f"Target Obsidian Vault: {vault_root}")
    if not vault_root.is_dir():
        print(f"Error: Vault path {vault_root} is not a directory or does not exist. Please check your configuration.")
        try:
            print(f"Attempting to create vault directory: {vault_root}")
            vault_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Could not create vault directory {vault_root}: {e}")
            exit(1)

    db_conn = None
    if duckdb:
        db_path = vault_root / config.get("database", {}).get("name", "obsidian_embeddings.db")
        try:
            print(f"Initializing DuckDB at: {db_path}")
            db_config = {
                'allow_unsigned_extensions': 'true',
                'autoinstall_known_extensions': 'true', 
                'autoload_known_extensions': 'true'    
            }
            # First, establish the connection
            db_conn = duckdb.connect(
                database=str(db_path), 
                read_only=False,
                config=db_config
            )
            
            # Attempt to explicitly install and load VSS after connection as a safeguard
            # The config flags should ideally handle this during connection for WAL replay.
            try:
                print("[DB Setup] Attempting to INSTALL 'vss' extension...")
                db_conn.execute("INSTALL 'vss';")
                print("[DB Setup] Attempting to LOAD 'vss' extension...")
                db_conn.execute("LOAD 'vss';")
                print("[DB Setup] 'vss' extension installed and loaded successfully.")
            except Exception as e:
                print(f"[DB Setup] Warning: Could not explicitly install/load 'vss' extension after connection: {e}. This might be okay if autoload worked.")
            
            print(f"DuckDB connection established. Autoloading flags used, and explicit VSS load attempted: {db_path}")

        except Exception as e:
            print(f"Error initializing DuckDB: {e}")
            if db_conn: 
                db_conn.close()
            db_conn = None
    else:
        print("DuckDB not available. Some features requiring a database will be disabled.")

    kv_store = None
    if TinyDB:
        kv_store_path = vault_root / config.get("kv_store", {}).get("name", "agent_kv_log.json")
        try:
            print(f"Initializing TinyDB at: {kv_store_path}")
            kv_store = TinyDB(kv_store_path, storage=CachingMiddleware(JSONStorage))
            print(f"TinyDB connection established: {kv_store_path}")
        except Exception as e: 
            print(f"Error initializing TinyDB at {kv_store_path}: {e}")
            kv_store = None
    else:
        print("TinyDB not available. Some features requiring a KV store will be disabled.")
        
    app_config = config.get("application_settings", {})

    embedding_model_config_for_ctx = config.get("embedding_model", {})
    final_embedding_dimensions = embedding_model_config_for_ctx.get("dimensions")
    if final_embedding_dimensions is None:
        final_embedding_dimensions = 1024 
        print(f"[VaultCtx Init] 'dimensions' not found in embedding_model config for VaultCtx, defaulting to {final_embedding_dimensions}.")
    else:
        print(f"[VaultCtx Init] Using 'dimensions': {final_embedding_dimensions} for VaultCtx.")

    return VaultCtx(
        root=vault_root,
        db=db_conn,
        kv=kv_store,
        config=app_config,
        embedding_dimensions=final_embedding_dimensions
    )

async def main_async_runner():
    """Main entry point for the Obsidian Agent runner."""
    parser = argparse.ArgumentParser(description="Obsidian Watchdog Agent.")
    parser.add_argument(
        "--run-backlinker-all",
        action="store_true",
        help="Run the backlinker agent for all notes and then exit."
    )
    args = parser.parse_args()

    print("Starting Obsidian Agent...")
    config_file_path = os.environ.get("AGENT_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    print(f"Loading configuration from: {config_file_path}")
    config = load_configuration(config_file_path)

    vault_context = await initialize_vault_context(config)

    # Ensure DB schema is up-to-date immediately after context initialization
    if vault_context.db:
        print("[Main] Ensuring database schema is up to date...")
        schema_was_changed = _ensure_db_schema(vault_context.db, vault_context.embedding_dimensions)
        if schema_was_changed:
            print("[Main] Database schema was updated by main runner.")
        else:
            print("[Main] Database schema confirmed to be up to date by main runner.")
    else:
        # This case is already handled below, but good to be aware
        print("[Main] DB context not available, skipping schema check.")

    if not vault_context.db:
        print("CRITICAL: DuckDB context not available. Some agent functionalities might fail. Exiting.")
        return

    if args.run_backlinker_all:
        print("[CLI] --run-backlinker-all flag detected. Running batch backlinker...")
        if vault_context.db:
            # First, ensure the database is populated with current notes
            print("[CLI] Populating notes database before batch backlinking...")
            embed_client_config_for_batch = config.get("embedding_model", {})
            if embed_client_config_for_batch.get("provider") and embed_client_config_for_batch.get("model_name"):
                embed_client_for_batch = get_embedding_client(config)
                try:
                    await populate_notes_db(vault_context.db, vault_context.root, embed_client_for_batch, config)
                    print("[CLI] Notes database population complete for batch mode.")
                except Exception as e:
                    print(f"[CLI] Error during database population in batch mode: {e}. Proceeding with potentially incomplete data.")
            else:
                print("[CLI] Skipped DB population for batch mode: Embedding model provider or model_name not configured.")

            # Now run the backlinker for all notes
            await run_backlinker_for_all_notes(vault_context)
            print("[CLI] Batch backlinker process finished.")
        else:
            print("[CLI] Cannot run batch backlinker: Database context not available.")
        
        # Clean up and exit after running the batch backlinker
        print("Obsidian Agent is shutting down after batch backlinker run.")
        if vault_context.db:
            try:
                vault_context.db.close()
                print("DuckDB connection closed.")
            except Exception as e: 
                print(f"Error closing DuckDB connection: {e}")
        if vault_context.kv:
            try:
                vault_context.kv.close()
                print("TinyDB connection closed.")
            except Exception as e:
                print(f"Error closing TinyDB connection: {e}")
        return # Exit after batch processing

    # Attempt to populate the database if conditions are met (only if not running batch backlinker)
    embed_client_config = config.get("embedding_model", {})
    if vault_context.db and embed_client_config.get("provider") and embed_client_config.get("model_name"):
        print("[DB Population] Attempting to populate notes database...")
        embed_client = get_embedding_client(config)
        try:
            await populate_notes_db(vault_context.db, vault_context.root, embed_client, config)
            print("[DB Population] Notes database population attempt finished.")
        except RuntimeError as e:
            if "cannot be called when another loop is running" in str(e):
                print(f"[DB Population] Error during population: {e}. This was unexpected with the async refactor.")
            else: 
                print(f"[DB Population] Unhandled RuntimeError during population: {e}") # Consider if this should raise
                # raise # Optionally re-raise if this is critical
        except Exception as e:
            print(f"[DB Population] An unexpected error occurred during database population: {e}")
            # Depending on severity, you might want to exit or ensure db is handled correctly
            # For now, we'll let the main observer run, but the DB might be incomplete.
    elif vault_context.db:
        print("[DB Population] Skipped: Embedding model provider or model_name not configured in config.yaml.")
    # else: vault_context.db is None, already handled by the critical check above

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
            except Exception as e: 
                print(f"Error closing DuckDB connection: {e}")
        if vault_context.kv:
            try:
                vault_context.kv.close()
                print("TinyDB connection closed.")
            except Exception as e:
                print(f"Error closing TinyDB connection: {e}")

if __name__ == "__main__":
    asyncio.run(main_async_runner())