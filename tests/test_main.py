import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import unittest.mock # Add this import

import pytest
import pytest_asyncio # Add this import
from obsidian_watchdog.main import initialize_vault_context, DEFAULT_VAULT_PATH
from obsidian_watchdog.deps import VaultCtx


@pytest_asyncio.fixture # Use pytest_asyncio.fixture for async fixtures
async def mock_vault_ctx():
    # This fixture might be expanded later if specific VaultCtx behavior needs mocking
    return MagicMock(spec=VaultCtx)

@pytest_asyncio.fixture # Use pytest_asyncio.fixture
async def mock_duckdb_conn():
    conn = MagicMock()
    conn.execute = MagicMock()
    conn.close = MagicMock()
    return conn

@pytest_asyncio.fixture # Use pytest_asyncio.fixture
async def mock_tinydb_store():
    store = MagicMock()
    store.close = MagicMock()
    return store

@pytest.mark.asyncio
async def test_initialize_vault_context_happy_path(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests the happy path for initialize_vault_context:
    - Vault path exists.
    - DuckDB initializes successfully.
    - TinyDB initializes successfully.
    - populate_notes_db is NOT called by this function (moved to main_async_runner).
    - VaultCtx is created with expected parameters.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimensions": 1536
        },
        "application_settings": {"some_setting": "value"}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        # Setup Path mock
        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_path_cls.return_value = mock_vault_root
        
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_path_cls.assert_called_once_with("/fake/vault")
        mock_vault_root.is_dir.assert_called_once()
        
        mock_duckdb_module.connect.assert_called_once_with(
            database=str(mock_vault_root / "test_obsidian.db"),
            read_only=False,
            config={
                "allow_unsigned_extensions": "true",
                "autoinstall_known_extensions": "true",
                "autoload_known_extensions": "true",
            }
        )
        mock_duckdb_conn.execute.assert_any_call("INSTALL 'vss';")
        mock_duckdb_conn.execute.assert_any_call("LOAD 'vss';")

        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()

        mock_tinydb_cls.assert_called_once_with(
            mock_vault_root / "test_agent_kv.json",
            storage=unittest.mock.ANY 
        )

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=mock_tinydb_store,
            config=mock_config["application_settings"],
            embedding_dimensions=1536
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_default_vault_path(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that the default vault path is used when not specified in config or environment.
    """
    mock_config = {
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimensions": 1536
        },
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls, \
         patch.dict(os.environ, {}, clear=True): # Ensure OBSIDIAN_VAULT_PATH is not set

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_path_cls.return_value = mock_vault_root
        
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        await initialize_vault_context(mock_config)

        mock_path_cls.assert_called_once_with(DEFAULT_VAULT_PATH)
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()
        # Further assertions similar to happy path can be added if needed

@pytest.mark.asyncio
async def test_initialize_vault_context_env_vault_path(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that vault path from environment variable is used when not in config.
    """
    env_vault_path = "/env/vault/path"
    mock_config = {
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimensions": 1536
        },
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls, \
         patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": env_vault_path}, clear=True):

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_path_cls.return_value = mock_vault_root

        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        await initialize_vault_context(mock_config)

        mock_path_cls.assert_called_once_with(env_vault_path)
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()
        # Further assertions

@pytest.mark.asyncio
async def test_initialize_vault_context_vault_path_creation_success(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that the vault path is created if it doesn't exist.
    """
    mock_config = {
        "vault_path": "/new/vault/to_create",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimensions": 1536
        }
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        # First call to is_dir returns False, second (after creation) True
        mock_vault_root.is_dir.side_effect = [False, True] 
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_vault_root.mkdir = MagicMock()
        mock_path_cls.return_value = mock_vault_root
        
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        await initialize_vault_context(mock_config)

        mock_path_cls.assert_called_once_with("/new/vault/to_create")
        assert mock_vault_root.is_dir.call_count == 1 # Should be 1, as mkdir is called before re-check in current code
        mock_vault_root.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()

@pytest.mark.asyncio
async def test_initialize_vault_context_vault_path_creation_failure(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that the application exits if vault path creation fails.
    """
    mock_config = {"vault_path": "/uncreatable/vault"}

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("builtins.exit") as mock_exit: # Patch builtins.exit

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = False
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_vault_root.mkdir.side_effect = OSError("Permission denied")
        mock_path_cls.return_value = mock_vault_root
        
        # No need to mock db/kv stores as exit should be called before

        await initialize_vault_context(mock_config)

        mock_path_cls.assert_called_once_with("/uncreatable/vault")
        mock_vault_root.is_dir.assert_called_once()
        mock_vault_root.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_exit.assert_called_once_with(1)

@pytest.mark.asyncio
async def test_initialize_vault_context_duckdb_not_available(mock_tinydb_store):
    """
    Tests behavior when duckdb module is not available (mocked as None).
    VaultCtx should be created with db=None.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "kv_store": {"name": "test_agent_kv.json"}, # kv_store still configured
        "embedding_model": {"dimensions": 512}, # Dimensions still configured
        "application_settings": {"setting": "val"}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb", None), \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_path_cls.return_value = mock_vault_root

        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_populate_notes_db.assert_not_called() # Should not be called if duckdb is None
        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=None, # Crucial assertion
            kv=mock_tinydb_store,
            config=mock_config["application_settings"],
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_duckdb_connect_fails(mock_tinydb_store):
    """
    Tests behavior when duckdb.connect raises an exception.
    VaultCtx should be created with db=None.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"dimensions": 512},
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_vault_root.expanduser.return_value = mock_vault_root
        mock_vault_root.resolve.return_value = mock_vault_root
        mock_path_cls.return_value = mock_vault_root

        mock_duckdb_module.connect.side_effect = Exception("DB Connection Error")
        
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_duckdb_module.connect.assert_called_once()
        mock_populate_notes_db.assert_not_called()
        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=None,
            kv=mock_tinydb_store,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_duckdb_vss_fails(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests behavior when DuckDB VSS extension loading fails.
    DB connection should still be passed to VaultCtx, and populate_notes_db should still be attempted.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"provider": "a", "model_name": "b", "dimensions": 512},
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root

        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        # Simulate VSS load failure
        mock_duckdb_conn.execute.side_effect = [None, Exception("VSS Load Fail")] 

        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_embed_client = AsyncMock()
        mock_get_embedding_client.return_value = mock_embed_client
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_duckdb_module.connect.assert_called_once()
        mock_duckdb_conn.execute.assert_any_call("INSTALL 'vss';")
        mock_duckdb_conn.execute.assert_any_call("LOAD 'vss';")
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()
        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn, # DB conn should still be passed
            kv=mock_tinydb_store,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_skip_populate_db_missing_embed_config(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that populate_notes_db is skipped if embedding model provider or name is missing.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"dimensions": 512}, # Missing provider and model_name
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()
        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=mock_tinydb_store,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_populate_db_runtime_error(mock_duckdb_conn, mock_tinydb_store):
    """
    Verifies that initialize_vault_context does not call populate_notes_db or get_embedding_client.
    This test previously checked handling of a RuntimeError from populate_notes_db within initialize_vault_context,
    which is no longer relevant as populate_notes_db is called by main_async_runner.
    It now ensures VaultCtx is created correctly with the db connection if DuckDB setup was otherwise successful.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"provider": "a", "model_name": "b", "dimensions": 512},
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root

        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance
        
        vault_context = await initialize_vault_context(mock_config)

        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn, 
            kv=mock_tinydb_store,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_populate_db_generic_error_sets_db_none(mock_tinydb_store):
    """
    Verifies that initialize_vault_context does not call populate_notes_db or get_embedding_client.
    This test previously checked if a generic error during populate_notes_db (when called by 
    initialize_vault_context) would lead to VaultCtx having db=None. This specific scenario is no longer applicable.
    If duckdb.connect() itself fails, db will be None (covered by test_initialize_vault_context_duckdb_connect_fails).
    If duckdb.connect() (mocked here by mock_inner_duckdb_conn) succeeds, and other internal DuckDB setup
    in initialize_vault_context (like VSS loading) proceeds without causing the entire block to fail and set db_conn to None,
    then the obtained db connection is passed to VaultCtx.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"provider": "a", "model_name": "b", "dimensions": 512}, 
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root

        mock_inner_duckdb_conn = MagicMock() 
        mock_inner_duckdb_conn.execute = MagicMock() 
        mock_inner_duckdb_conn.close = MagicMock()
        mock_duckdb_module.connect.return_value = mock_inner_duckdb_conn
        
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)
        
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()
        mock_inner_duckdb_conn.close.assert_not_called() 

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_inner_duckdb_conn, 
            kv=mock_tinydb_store,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_tinydb_not_available(mock_duckdb_conn):
    """
    Tests behavior when TinyDB module is not available.
    VaultCtx should be created with kv=None.
    populate_notes_db and get_embedding_client are not called by this function.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"}, 
        "embedding_model": {"provider": "a", "model_name": "b", "dimensions": 512}, 
        "application_settings": {"setting": "val"}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB", None), \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root

        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=None, 
            config=mock_config["application_settings"],
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_tinydb_init_fails(mock_duckdb_conn):
    """
    Tests behavior when TinyDB() constructor raises an exception.
    VaultCtx should be created with kv=None.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"provider": "a", "model_name": "b", "dimensions": 512},
        "application_settings": {}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock), \
         patch("obsidian_watchdog.main.get_embedding_client"), \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root

        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.side_effect = Exception("TinyDB Init Error")
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_tinydb_cls.assert_called_once()
        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=None,
            config={},
            embedding_dimensions=512
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_default_embedding_dimensions(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests that default embedding dimensions (1024) are used if not specified in config.
    """
    mock_config = {
        "vault_path": "/fake/vault",
        "database": {"name": "test_obsidian.db"},
        "kv_store": {"name": "test_agent_kv.json"},
        "embedding_model": {"provider": "a", "model_name": "b"}, # Dimensions missing
        "application_settings": {"app_setting": "app_value"}
    }

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock), \
         patch("obsidian_watchdog.main.get_embedding_client"), \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls:

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(mock_config)

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=mock_tinydb_store,
            config=mock_config["application_settings"],
            embedding_dimensions=1024 # Default value
        )
        assert vault_context == mock_vault_ctx_instance

@pytest.mark.asyncio
async def test_initialize_vault_context_empty_config(mock_duckdb_conn, mock_tinydb_store):
    """
    Tests behavior when an empty config is passed (e.g., from load_configuration due to file not found or parse error).
    Default values should be used for paths, db names, kv names, dimensions, etc.
    """
    empty_config = {}

    with patch("obsidian_watchdog.main.Path") as mock_path_cls, \
         patch("obsidian_watchdog.main.duckdb") as mock_duckdb_module, \
         patch("obsidian_watchdog.main.TinyDB") as mock_tinydb_cls, \
         patch("obsidian_watchdog.main.populate_notes_db", new_callable=AsyncMock) as mock_populate_notes_db, \
         patch("obsidian_watchdog.main.get_embedding_client") as mock_get_embedding_client, \
         patch("obsidian_watchdog.main.VaultCtx") as mock_vault_ctx_cls, \
         patch.dict(os.environ, {}, clear=True): # Ensure no env var for vault path

        mock_vault_root = MagicMock(spec=Path)
        mock_vault_root.is_dir.return_value = True
        mock_path_cls.return_value.expanduser.return_value.resolve.return_value = mock_vault_root
        
        mock_duckdb_module.connect.return_value = mock_duckdb_conn
        mock_tinydb_cls.return_value = mock_tinydb_store
        mock_vault_ctx_instance = MagicMock(spec=VaultCtx)
        mock_vault_ctx_cls.return_value = mock_vault_ctx_instance

        vault_context = await initialize_vault_context(empty_config)

        mock_path_cls.assert_called_once_with(DEFAULT_VAULT_PATH)
        mock_duckdb_module.connect.assert_called_once_with(
            database=str(mock_vault_root / "obsidian_embeddings.db"), # Default DB name
            read_only=False,
            config={
                "allow_unsigned_extensions": "true",
                "autoinstall_known_extensions": "true",
                "autoload_known_extensions": "true",
            }
        )
        mock_get_embedding_client.assert_not_called()
        mock_populate_notes_db.assert_not_called()

        mock_tinydb_cls.assert_called_once_with(
            mock_vault_root / "agent_kv_log.json", # Default KV name
            storage=unittest.mock.ANY 
        )

        mock_vault_ctx_cls.assert_called_once_with(
            root=mock_vault_root,
            db=mock_duckdb_conn,
            kv=mock_tinydb_store,
            config={}, # Default app_config
            embedding_dimensions=1024 # Default dimensions
        )
        assert vault_context == mock_vault_ctx_instance

# We will add more tests here for other scenarios 