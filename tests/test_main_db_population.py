import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import duckdb
import pytest
import pytest_asyncio # Required for async fixtures
import yaml
from openai import AsyncOpenAI # For type hinting the mock

# Assuming DB_SCHEMA_VERSION and populate_notes_db are in src.obsidian_watchdog.main
# Adjust the import path if your structure is different or if you move these.
from src.obsidian_watchdog.notes_db_manager import DB_SCHEMA_VERSION, populate_notes_db

# Default embedding dimension, matching the one in main.py if not in config
DEFAULT_EMBEDDING_DIMENSION = 1024

@pytest.fixture
def base_config() -> dict:
    """Provides a base configuration dictionary for tests."""
    return {
        "embedding_model": {
            "model_name": "test-embedding-model",
            "dimensions": DEFAULT_EMBEDDING_DIMENSION,
            "provider": "test_provider" # Added to prevent skipping db population
        },
        "db_population": {
            "ignore_patterns": [".obsidian/", "ai_logs/"],
            "force_reembed_all": False,
            "embedding_batch_size": 10,
        },
        "database": {
            "name": "test_obsidian_embeddings.db" # Though we use in-memory
        }
    }

@pytest_asyncio.fixture
async def mock_embed_client(base_config: dict):
    """Mocks the AsyncOpenAI client for embedding generation."""
    client = AsyncMock(spec=AsyncOpenAI)
    client.embeddings = MagicMock()

    # Determine embedding dimension from config
    dim = base_config["embedding_model"].get("dimensions", DEFAULT_EMBEDDING_DIMENSION)

    # This function will be called by the mock
    async def mock_create_embeddings(*args, input, model, dimensions=None, **kwargs):
        # Print for debugging test calls
        # print(f"Mock embed_client.create called with input: {input}, model: {model}, dimensions: {dimensions}")
        response = MagicMock()
        response.data = []
        if isinstance(input, str): # Single input
            input_list = [input]
        else: # List of inputs
            input_list = input
        
        for i, _ in enumerate(input_list):
            embedding_item = MagicMock()
            embedding_item.embedding = [0.01 * (i + 1)] * dim # Create unique-ish embeddings
            embedding_item.index = i
            embedding_item.object = "embedding"
            response.data.append(embedding_item)
        return response

    client.embeddings.create = AsyncMock(side_effect=mock_create_embeddings)
    return client

@pytest_asyncio.fixture
async def temp_duckdb_conn():
    """Provides an in-memory DuckDB connection for tests."""
    conn = duckdb.connect(':memory:')
    try:
        # print("[Test DB] Attempting to install and load 'vss' extension...")
        conn.execute("INSTALL 'vss';")
        conn.execute("LOAD 'vss';")
        # print("[Test DB] 'vss' extension loaded.")
        conn.execute("SET TimeZone='UTC';") # Ensure session timezone is UTC
        # print("[Test DB] Session timezone set to UTC.")
    except Exception as e:
        print(f"[Test DB] Error loading 'vss' extension or setting timezone: {e}")
        # Depending on how critical VSS is for the tests, you might raise or skip.
        # For populate_notes_db, HNSW index creation relies on it.
        pytest.fail(f"Failed to load VSS extension for DuckDB: {e}")
    
    yield conn # Provide the connection to the test
    
    # print("[Test DB] Closing in-memory DuckDB connection.")
    conn.close()

def create_mock_file(vault_path: Path, relative_path: str, content: str, m_time: datetime = None):
    """Helper to create a mock markdown file in the temporary vault."""
    full_path = vault_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    # if m_time:
    #     # Convert datetime to timestamp for os.utime
    #     timestamp = m_time.timestamp()
    #     os.utime(full_path, (timestamp, timestamp))
    return full_path

@pytest.mark.asyncio
async def test_initial_population_empty_db(
    temp_duckdb_conn: duckdb.DuckDBPyConnection, # type: ignore
    mock_embed_client: AsyncMock, # type: ignore
    tmp_path: Path, # Pytest fixture for temporary directory
    base_config: dict
):
    """
    Tests initial population of an empty database with a few new files.
    """
    db = temp_duckdb_conn
    vault_root = tmp_path / "vault"
    vault_root.mkdir()

    # Create some mock files
    file1_content = "This is note 1."
    file1_rel_path = "Note1.md"
    # Create the file first
    file1_path = create_mock_file(vault_root, file1_rel_path, file1_content)
    # Then get its actual mtime as the reference
    file1_mtime_ts = file1_path.stat().st_mtime
    file1_mtime = datetime.fromtimestamp(file1_mtime_ts, tz=timezone.utc)

    file2_content = "## Heading for Note 2\nSome details."
    file2_rel_path = "subfolder/Note2.md"
    # Create the file first
    file2_path = create_mock_file(vault_root, file2_rel_path, file2_content)
    # Then get its actual mtime as the reference
    file2_mtime_ts = file2_path.stat().st_mtime
    file2_mtime = datetime.fromtimestamp(file2_mtime_ts, tz=timezone.utc)

    # --- Act ---
    await populate_notes_db(db, vault_root, mock_embed_client, base_config)

    # --- Assert ---
    # 1. Check DB version
    version_info = db.execute("SELECT version FROM db_version ORDER BY version DESC LIMIT 1").fetchone()
    assert version_info is not None, "db_version table should have an entry"
    assert version_info[0] == DB_SCHEMA_VERSION, "DB schema version should match"

    # 2. Check notes table structure and content
    notes_data = db.execute("SELECT path, content_hash, modified_at, last_embedded_at, raw_content, embedding FROM notes ORDER BY path").fetchall()
    assert len(notes_data) == 2, "Should be two notes in the database"

    # Note 1 assertions
    note1_db = notes_data[0]
    assert note1_db[0] == file1_rel_path.replace("\\", "/") # Normalize path separator for comparison
    expected_hash1 = hashlib.md5(file1_content.encode('utf-8')).hexdigest()
    assert note1_db[1] == expected_hash1
    # DuckDB stores TIMESTAMP WITH TIME ZONE as UTC.
    assert note1_db[2] == file1_mtime 
    assert note1_db[3] is not None and isinstance(note1_db[3], datetime) # last_embedded_at
    # Ensure it's timezone-aware (should be UTC from TIMESTAMP WITH TIME ZONE)
    assert note1_db[3].tzinfo is not None and note1_db[3].tzinfo.utcoffset(note1_db[3]) == timedelta(0)
    assert note1_db[4] == file1_content
    assert len(note1_db[5]) == base_config["embedding_model"]["dimensions"] # Check embedding vector length
    assert note1_db[5][0] != 0.0 # Check if embedding seems populated (mock generates non-zero)

    # Note 2 assertions
    note2_db = notes_data[1]
    assert note2_db[0] == file2_rel_path.replace("\\", "/")
    expected_hash2 = hashlib.md5(file2_content.encode('utf-8')).hexdigest()
    assert note2_db[1] == expected_hash2
    assert note2_db[2] == file2_mtime
    assert note2_db[3] is not None and isinstance(note2_db[3], datetime)
    assert note2_db[3].tzinfo is not None and note2_db[3].tzinfo.utcoffset(note2_db[3]) == timedelta(0)
    assert note2_db[4] == file2_content
    assert len(note2_db[5]) == base_config["embedding_model"]["dimensions"]

    # 3. Check embedding client calls
    # Since batch_size is 10 and we have 2 notes, it should be called once for the batch.
    assert mock_embed_client.embeddings.create.call_count == 1
    call_args = mock_embed_client.embeddings.create.call_args
    assert sorted(call_args.kwargs['input']) == sorted([file1_content, file2_content])
    assert call_args.kwargs['model'] == base_config["embedding_model"]["model_name"]
    assert call_args.kwargs['dimensions'] == base_config["embedding_model"]["dimensions"]
    
    # 4. Check HNSW index
    # DuckDB stores index info in duckdb_indexes()
    # This is a basic check; more advanced would be to ensure it's usable.
    index_info = db.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name='notes' AND index_name='hnsw_embedding_idx'").fetchone()
    assert index_info is not None, "HNSW index 'hnsw_embedding_idx' should exist on 'notes' table"

    # Check diagnostic for table structure (embedding column type)
    table_info = db.execute("SELECT column_name, data_type FROM duckdb_columns() WHERE table_name = 'notes' AND column_name = 'embedding'").fetchone()
    assert table_info is not None
    expected_embedding_type = f"FLOAT[{base_config['embedding_model']['dimensions']}]"
    assert table_info[1] == expected_embedding_type, f"Embedding column type should be {expected_embedding_type}"

# To run this test:
# 1. Make sure pytest, pytest-asyncio, pytest-mock are installed in your .venv
#    uv pip install -d pytest pytest-asyncio pytest-mock
# 2. Navigate to your project root in the terminal
# 3. Run: pytest tests/test_main_db_population.py 