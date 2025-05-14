import hashlib
from datetime import datetime, timezone
from pathlib import Path
import duckdb # For DuckDBPyConnection type hint
from openai import AsyncOpenAI # For AsyncOpenAI type hint

# Moved from main.py
DB_SCHEMA_VERSION = 7 # Incremented for backlinker metadata support

def _ensure_db_schema(db: duckdb.DuckDBPyConnection, embedding_dimensions: int) -> bool:
    """Ensures the database schema is up to date. Creates or migrates the notes table."""
    db.execute("CREATE TABLE IF NOT EXISTS db_version (version INTEGER PRIMARY KEY)")
    current_db_version_row = db.execute("SELECT version FROM db_version ORDER BY version DESC LIMIT 1").fetchone()
    current_db_version = current_db_version_row[0] if current_db_version_row else 0

    if current_db_version < DB_SCHEMA_VERSION:
        print(f"[DB Schema] DB schema version mismatch (DB: {current_db_version}, Code: {DB_SCHEMA_VERSION}). Re-creating notes table.")
        db.execute("DROP TABLE IF EXISTS notes")
        create_table_sql = f"""
        CREATE TABLE notes (
            path VARCHAR NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_start INTEGER NOT NULL,
            chunk_end INTEGER NOT NULL,
            content_hash VARCHAR NOT NULL,
            modified_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_embedded_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_backlinked_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
            raw_content TEXT,
            embedding FLOAT[{embedding_dimensions}],
            PRIMARY KEY (path, chunk_index)
        );
        """
        db.execute(create_table_sql)
        db.execute("INSERT OR REPLACE INTO db_version (version) VALUES (?)", [DB_SCHEMA_VERSION])
        print("[DB Schema] 'notes' table created/updated with new schema.")
        return True # Indicates schema was changed
    print("[DB Schema] Database schema is up to date.")
    return False # Indicates schema was not changed

def _get_existing_notes_metadata(db: duckdb.DuckDBPyConnection) -> dict:
    """Fetches metadata of existing notes (path, hash, last_embedded_at) from the database."""
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
        print(f"[DB Meta] Found {len(existing_notes_data)} existing note records in DB for comparison.")
    except duckdb.CatalogException: # type: ignore
        print("[DB Meta] 'notes' table not found. Proceeding as if empty (this might happen if schema was just created).")
    return existing_notes_data

def _scan_vault_for_notes(vault_root: Path, ignore_patterns: list[str]) -> list[dict]:
    """Scans the vault for markdown files, returning a list of their details."""
    disk_notes = []
    print(f"[Vault Scan] Scanning vault: {vault_root} (ignoring: {ignore_patterns}) for markdown files...")
    for filepath in vault_root.glob("**/*.md"):
        try:
            relative_path_str = str(filepath.relative_to(vault_root)).replace("\\", "/") # Normalize path separators
            if any(relative_path_str.startswith(pattern) for pattern in ignore_patterns):
                continue

            m_time_utc = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
            content = filepath.read_text(encoding="utf-8")
            current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            disk_notes.append({
                "path": relative_path_str,
                "content": content,
                "modified_at": m_time_utc,
                "hash": current_hash,
                "filepath": filepath 
            })
        except Exception as e:
            print(f"[Vault Scan] Error processing file {filepath} during scan: {e}")
    print(f"[Vault Scan] Found {len(disk_notes)} markdown files on disk.")
    return disk_notes

def _identify_notes_for_processing(disk_notes: list[dict], existing_db_notes_metadata: dict, force_reembed_all: bool) -> list[dict]:
    """Identifies notes that need to be processed (new, modified, or force re-embed)."""
    notes_to_process = []
    print("[Note Identification] Identifying notes to process...")
    for note_info in disk_notes:
        relative_path_str = note_info["path"]
        current_hash = note_info["hash"]
        m_time_utc = note_info["modified_at"]
        
        existing_data = existing_db_notes_metadata.get(relative_path_str)
        should_process = False
        reason = ""

        if force_reembed_all:
            should_process = True
            reason = "force_reembed_all is true"
        elif not existing_data:
            should_process = True
            reason = "new note"
        elif existing_data["hash"] != current_hash:
            should_process = True
            reason = "content changed (hash mismatch)"
        elif existing_data.get("embedded_at") is None: 
            should_process = True
            reason = "missing previous embedding timestamp"
        elif m_time_utc > existing_data["embedded_at"]:
            should_process = True
            reason = "note modified after last embedding"
        
        if should_process:
            print(f"[Note Identification] Queuing '{relative_path_str}' for processing. Reason: {reason}.")
            notes_to_process.append({
                "path": relative_path_str,
                "content": note_info["content"], 
                "modified_at": m_time_utc,
                "hash": current_hash
            })
    print(f"[Note Identification] Identified {len(notes_to_process)} notes for embedding/re-embedding.")
    return notes_to_process

async def _batch_embed_and_store_notes(
    db: duckdb.DuckDBPyConnection, 
    notes_to_process: list[dict], 
    embed_client: AsyncOpenAI, 
    config: dict, 
    embedding_api_model_name: str, 
    embedding_dimensions: int 
):
    """Generates embeddings in batches and stores them in the database. Now supports chunked notes."""
    if not notes_to_process:
        print("[Embedding] No notes to process for embedding.")
        return

    print(f"[Embedding] Found {len(notes_to_process)} notes to generate/update embeddings for.")
    population_config = config.get("db_population", {})
    batch_size = population_config.get("embedding_batch_size", 50)
    chunk_size = population_config.get("embedding_chunk_size", 1000)  # Number of characters per chunk (default 1000)

    # Step 1: Prepare all chunks from all notes
    all_chunks = []
    for note in notes_to_process:
        content = note["content"]
        path = note["path"]
        hash_ = note["hash"]
        modified_at = note["modified_at"]
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunks.append({
                "path": path,
                "chunk_index": chunk_index,
                "chunk_start": start,
                "chunk_end": end,
                "content_hash": hash_,
                "modified_at": modified_at,
                "raw_content": chunk_text
            })
            start = end
            chunk_index += 1
        all_chunks.extend(chunks)

    print(f"[Embedding] Prepared {len(all_chunks)} total chunks from {len(notes_to_process)} notes.")

    # Step 2: Embed and store in batches
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        contents_to_embed = [chunk["raw_content"] for chunk in batch]
        print(f"[Embedding] Generating embeddings for chunk batch {i // batch_size + 1} of {(len(all_chunks) + batch_size - 1) // batch_size} (size: {len(batch)}). Model: {embedding_api_model_name}...")
        try:
            embedding_params = {"input": contents_to_embed, "model": embedding_api_model_name}
            if embedding_dimensions: 
                embedding_params["dimensions"] = embedding_dimensions
            raw_embeddings_response = await embed_client.embeddings.create(**embedding_params)
            embeddings_vectors = [item.embedding for item in raw_embeddings_response.data]

            db_batch_data = []
            now = datetime.now(timezone.utc)
            for chunk, emb_vector in zip(batch, embeddings_vectors):
                db_batch_data.append((
                    chunk["path"],
                    chunk["chunk_index"],
                    chunk["chunk_start"],
                    chunk["chunk_end"],
                    chunk["content_hash"],
                    chunk["modified_at"],
                    now,
                    chunk["raw_content"],
                    emb_vector
                ))
            db.executemany(
                """
                INSERT INTO notes (path, chunk_index, chunk_start, chunk_end, content_hash, modified_at, last_embedded_at, last_backlinked_at, raw_content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(path, chunk_index) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    modified_at=excluded.modified_at,
                    last_embedded_at=excluded.last_embedded_at,
                    raw_content=excluded.raw_content,
                    embedding=excluded.embedding,
                    chunk_start=excluded.chunk_start,
                    chunk_end=excluded.chunk_end
                """,
                db_batch_data
            )
            print(f"[Embedding] Successfully inserted/updated chunk batch {i // batch_size + 1} into DB.")
        except Exception as e:
            print(f"[Embedding] Error generating embeddings or inserting chunk batch {i // batch_size + 1}: {e}")

def _prune_deleted_notes(db: duckdb.DuckDBPyConnection, vault_root: Path, ignore_patterns: list[str], existing_db_notes_paths: list[str]):
    """Prunes notes from the database that no longer exist in the vault filesystem."""
    print("[Pruning] Starting pruning check for deleted notes...")
    all_disk_paths_set = set()
    for fp in vault_root.glob("**/*.md"):
        try:
            relative_path = str(fp.relative_to(vault_root)).replace("\\", "/")
            if not any(relative_path.startswith(p) for p in ignore_patterns):
                all_disk_paths_set.add(relative_path)
        except Exception as e:
            print(f"[Pruning] Error processing file {fp} during disk scan for pruning: {e}")

    db_paths_to_prune = [path for path in existing_db_notes_paths if path not in all_disk_paths_set]
    
    if db_paths_to_prune:
        print(f"[Pruning] Pruning {len(db_paths_to_prune)} notes from DB that no longer exist on disk...")
        for path_to_prune in db_paths_to_prune:
            try:
                db.execute("DELETE FROM notes WHERE path = ?", [path_to_prune])
            except Exception as e:
                print(f"[Pruning] Error deleting note {path_to_prune} from DB: {e}")
        print("[Pruning] Pruning complete.")
    else:
        print("[Pruning] No notes found to prune from the database.")

def _ensure_hnsw_index(db: duckdb.DuckDBPyConnection, embedding_dimensions: int):
    """Ensures the HNSW index for note embeddings exists and is up-to-date."""
    print("[HNSW Index] Attempting to create/update HNSW index for note embeddings...")
    try:
        db.execute("SET hnsw_enable_experimental_persistence = true;")

        print("[HNSW Index DIAGNOSTIC] Checking 'notes' table structure before creating HNSW index...")
        try:
            table_info = db.execute("SELECT column_name, data_type FROM duckdb_columns() WHERE table_name = 'notes' AND column_name = 'embedding'").fetchone()
            if table_info:
                print(f"[HNSW Index DIAGNOSTIC] 'embedding' column type: {table_info[1]}")
                expected_type = f'FLOAT[{embedding_dimensions}]'
                if table_info[1] != expected_type:
                    print(f"[HNSW Index DIAGNOSTIC] MISMATCH! Expected {expected_type}, got {table_info[1]}")
            else:
                print("[HNSW Index DIAGNOSTIC] 'embedding' column not found in 'notes' table details.")
        except Exception as diag_e:
            print(f"[HNSW Index DIAGNOSTIC] Error fetching table structure: {diag_e}")
        
        db.execute("""
        CREATE INDEX IF NOT EXISTS hnsw_embedding_idx 
        ON notes USING HNSW (embedding) 
        WITH (metric = 'cosine', M = 16, ef_construction = 100);
        """) 
        print("[HNSW Index] HNSW index 'hnsw_embedding_idx' on 'notes.embedding' ensured.")
    except Exception as e:
        print(f"[HNSW Index] Warning: Could not create/update HNSW index: {e}")
        if "FLOAT[N]" in str(e) or "FLOAT[" in str(e): # Broader check for the error message
            print(f"[HNSW Index] Hint: This error often means the 'embedding_dimensions' ({embedding_dimensions}) in your config or code does not match the actual output dimension of your embedding model, or the table schema needs to be updated to FLOAT[{embedding_dimensions}].")
        print("[HNSW Index] Hint: Ensure the VSS extension is correctly loaded and DuckDB version supports these features.")

def _run_post_population_diagnostics(db: duckdb.DuckDBPyConnection):
    """Runs diagnostic checks after the population process."""
    try:
        print("[Post-Pop DIAGNOSTIC] Checking for 'Data Insights Findings.md' post-population...")
        specific_note_check = db.execute("SELECT path, embedding IS NOT NULL AS has_embedding, array_length(embedding) as embedding_len FROM notes WHERE path = 'Data Insights Findings.md'").fetchone()
        if specific_note_check:
            print(f"[Post-Pop DIAGNOSTIC] Found 'Data Insights Findings.md': Path='{specific_note_check[0]}', HasEmbedding='{specific_note_check[1]}', EmbeddingLength='{specific_note_check[2]}'")
        else:
            print("[Post-Pop DIAGNOSTIC] 'Data Insights Findings.md' NOT FOUND in notes table after population.")
        
        total_notes_count = db.execute("SELECT COUNT(*) FROM notes").fetchone()
        print(f"[Post-Pop DIAGNOSTIC] Total notes in DB after population: {total_notes_count[0] if total_notes_count else 'Error'}")
        
        sample_paths = db.execute("SELECT path FROM notes LIMIT 5").fetchall()
        print(f"[Post-Pop DIAGNOSTIC] Sample paths in DB: {sample_paths}")

    except Exception as diag_e_post:
        print(f"[Post-Pop DIAGNOSTIC] Error during post-population check: {diag_e_post}")

async def populate_notes_db(db: duckdb.DuckDBPyConnection, vault_root: Path, embed_client: AsyncOpenAI, config: dict): # type: ignore
    """Scans the vault, generates embeddings, and populates the DuckDB notes table using helper functions."""
    print("[DB Population] Starting to populate/update notes database...")

    embedding_model_config = config.get("embedding_model", {})
    embedding_api_model_name = embedding_model_config.get("model_name", "text-embedding-mxbai-embed-large-v1")
    embedding_dimensions = embedding_model_config.get("dimensions")

    if embedding_dimensions is None:
        embedding_dimensions = 1024 
        print(f"[DB Population] 'dimensions' not found in embedding_model config, defaulting to {embedding_dimensions}. Verify this matches your model's output.")
    else:
        print(f"[DB Population] Using 'dimensions': {embedding_dimensions} from config for embedding schema, HNSW index, and API calls.")

    schema_changed = _ensure_db_schema(db, embedding_dimensions)
    existing_notes_metadata = {} if schema_changed else _get_existing_notes_metadata(db)

    population_config = config.get("db_population", {})
    ignore_patterns = population_config.get("ignore_patterns", [".obsidian/", ".git/", "ai_logs/", "node_modules/"])
    disk_notes = _scan_vault_for_notes(vault_root, ignore_patterns)

    force_reembed_all = population_config.get("force_reembed_all", False)
    if force_reembed_all:
        print("[DB Population] force_reembed_all is TRUE. All notes will be re-embedded.")
    notes_to_process = _identify_notes_for_processing(disk_notes, existing_notes_metadata, force_reembed_all)

    if notes_to_process:
        await _batch_embed_and_store_notes(
            db, 
            notes_to_process, 
            embed_client, 
            config, 
            embedding_api_model_name, 
            embedding_dimensions
        )
    else:
        print("[DB Population] No new or modified notes to process for embedding based on current checks.")

    current_db_paths = list(existing_notes_metadata.keys())
    _prune_deleted_notes(db, vault_root, ignore_patterns, current_db_paths)
    _ensure_hnsw_index(db, embedding_dimensions)
    _run_post_population_diagnostics(db)

    print("[DB Population] Database population/update process finished.") 