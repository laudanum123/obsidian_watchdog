import pytest
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.obsidian_watchdog.deps import VaultCtx, RECENTLY_MODIFIED_TTL_SECONDS

@pytest.fixture
def mock_vault_root() -> Path:
    return MagicMock(spec=Path)

@pytest.fixture
def mock_duckdb_conn():
    return MagicMock()

@pytest.fixture
def mock_tinydb():
    return MagicMock()

@pytest.fixture
def vault_ctx(mock_vault_root, mock_duckdb_conn, mock_tinydb) -> VaultCtx:
    return VaultCtx(
        root=mock_vault_root,
        db=mock_duckdb_conn,
        kv=mock_tinydb,
        config={},
        embedding_dimensions=1024
    )

def test_add_agent_modified_path(vault_ctx: VaultCtx):
    test_path = "/fake/path/file.md"
    with patch('time.monotonic', return_value=100.0):
        vault_ctx.add_agent_modified_path(test_path)
    assert test_path in vault_ctx._agent_modified_paths
    assert vault_ctx._agent_modified_paths[test_path] == 100.0

def test_was_recently_modified_by_agent_true(vault_ctx: VaultCtx):
    test_path = "/fake/path/file.md"
    current_time = 100.0
    
    with patch('time.monotonic') as mock_time_monotonic:
        mock_time_monotonic.return_value = current_time
        vault_ctx.add_agent_modified_path(test_path) # Added at 100.0
        
        # Check immediately, should be true
        mock_time_monotonic.return_value = current_time + 1.0 # Time moves to 101.0
        assert vault_ctx.was_recently_modified_by_agent(test_path) is True

def test_was_recently_modified_by_agent_false_never_added(vault_ctx: VaultCtx):
    test_path = "/fake/path/never_added.md"
    with patch('time.monotonic', return_value=100.0):
        assert vault_ctx.was_recently_modified_by_agent(test_path) is False

def test_was_recently_modified_by_agent_false_expired(vault_ctx: VaultCtx):
    test_path = "/fake/path/expired.md"
    initial_time = 100.0
    
    with patch('time.monotonic') as mock_time_monotonic:
        mock_time_monotonic.return_value = initial_time
        vault_ctx.add_agent_modified_path(test_path) # Added at 100.0
        
        # Advance time beyond TTL
        mock_time_monotonic.return_value = initial_time + RECENTLY_MODIFIED_TTL_SECONDS + 1.0
        assert vault_ctx.was_recently_modified_by_agent(test_path) is False
        # Check that it was removed from the dictionary
        assert test_path not in vault_ctx._agent_modified_paths

def test_was_recently_modified_by_agent_cleanup_expired_entries(vault_ctx: VaultCtx):
    path1_fresh = "/fake/path/fresh.md"
    path2_expired = "/fake/path/expired.md"
    path3_other_fresh = "/fake/path/other_fresh.md"
    
    initial_time = 100.0
    
    with patch('time.monotonic') as mock_time_monotonic:
        # Add path2_expired at initial_time
        mock_time_monotonic.return_value = initial_time
        vault_ctx.add_agent_modified_path(path2_expired)
        
        # Add path1_fresh slightly later
        time_for_fresh_paths = initial_time + RECENTLY_MODIFIED_TTL_SECONDS / 2
        mock_time_monotonic.return_value = time_for_fresh_paths
        vault_ctx.add_agent_modified_path(path1_fresh)
        vault_ctx.add_agent_modified_path(path3_other_fresh)

        # Advance time so path2_expired is expired, but fresh ones are not
        check_time = initial_time + RECENTLY_MODIFIED_TTL_SECONDS + 1.0
        mock_time_monotonic.return_value = check_time
        
        # Calling was_recently_modified_by_agent for any path should trigger cleanup
        # Let's check a path that wasn't added, to ensure cleanup still runs
        vault_ctx.was_recently_modified_by_agent("/fake/path/non_existent.md")
        
        assert path1_fresh in vault_ctx._agent_modified_paths
        assert path3_other_fresh in vault_ctx._agent_modified_paths
        assert path2_expired not in vault_ctx._agent_modified_paths
        
        # Check actual status
        assert vault_ctx.was_recently_modified_by_agent(path1_fresh) is True
        assert vault_ctx.was_recently_modified_by_agent(path3_other_fresh) is True
        assert vault_ctx.was_recently_modified_by_agent(path2_expired) is False

def test_was_recently_modified_by_agent_empty_dict(vault_ctx: VaultCtx):
    with patch('time.monotonic', return_value=100.0):
        assert vault_ctx.was_recently_modified_by_agent("any/path") is False
        assert not vault_ctx._agent_modified_paths # Ensure dict remains empty

def test_was_recently_modified_by_agent_exact_ttl_edge_case(vault_ctx: VaultCtx):
    test_path = "/fake/path/edge.md"
    initial_time = 100.0

    with patch('time.monotonic') as mock_time_monotonic:
        mock_time_monotonic.return_value = initial_time
        vault_ctx.add_agent_modified_path(test_path) # Added at 100.0

        # Time is exactly at the TTL boundary.
        # current_time - ts > RECENTLY_MODIFIED_TTL_SECONDS will be FALSE
        # 100.0 + 5.0 (current_time) - 100.0 (ts) = 5.0.  5.0 > 5 is False.
        # So it should still be considered recent.
        mock_time_monotonic.return_value = initial_time + RECENTLY_MODIFIED_TTL_SECONDS
        assert vault_ctx.was_recently_modified_by_agent(test_path) is True
        assert test_path in vault_ctx._agent_modified_paths # Should not be deleted yet

        # Time is just past the TTL boundary
        # current_time - ts > RECENTLY_MODIFIED_TTL_SECONDS will be TRUE
        # 100.0 + 5.0 + 0.001 (current_time) - 100.0 (ts) = 5.001. 5.001 > 5 is True.
        # So it should be considered expired and removed.
        mock_time_monotonic.return_value = initial_time + RECENTLY_MODIFIED_TTL_SECONDS + 0.001
        assert vault_ctx.was_recently_modified_by_agent(test_path) is False
        assert test_path not in vault_ctx._agent_modified_paths # Should be deleted 