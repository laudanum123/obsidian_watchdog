import pytest
import asyncio
import shutil
from pathlib import Path
import pathlib
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock, AsyncMock
import builtins

# Imports from the project
# Assuming FsEvent and VaultCtx are defined in agents.backlinker or accessible from models
from obsidian_watchdog.agents.backlinker import run_backlinker_for_event, VaultCtx, FsEvent, _get_similar_note_paths_INTERNAL, LinkDecision
# from models import EditScriptAction # If needed for type hints or mocks, not directly used now

# Mock DB (adapted from agents/backlinker.py's test section)
class SimpleMockDb:
    def __init__(self, vault_root: Path, embedding_dim: int = 3):
        self.vault_root = vault_root
        self.embedding_dim = embedding_dim
        self.notes_embeddings: Dict[str, List[float]] = {}
        self._current_query: str = ""
        self._current_params: list = []

    def add_note_embedding(self, path: str, embedding: List[float]):
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding for {path} has dimension {len(embedding)}, expected {self.embedding_dim}")
        self.notes_embeddings[path] = embedding

    def execute(self, query: str, params: list = None):
        self._current_query = query
        self._current_params = params if params is not None else []
        return self

    def fetchone(self):
        if "SELECT embedding FROM notes WHERE path = ?" in self._current_query and self._current_params:
            path = self._current_params[0]
            embedding = self.notes_embeddings.get(path)
            return (embedding,) if embedding else None
        return None

    def fetchall(self):
        if "array_cosine_similarity" in self._current_query and len(self._current_params) >= 3:
            query_embedding = self._current_params[0]
            current_path = self._current_params[1]
            limit = self._current_params[2]

            if not query_embedding or len(query_embedding) != self.embedding_dim:
                # print(f"Warning: Query embedding is invalid or dimension mismatch. Dim: {self.embedding_dim}, Got: {query_embedding}")
                return []

            results = []
            for path, emb in self.notes_embeddings.items():
                if path == current_path:
                    continue
                if len(emb) != self.embedding_dim: # Should not happen if add_note_embedding validates
                    # print(f"Warning: Embedding for {path} dimension mismatch during similarity calc.")
                    continue
                
                # Pseudo-similarity: sum of products (not normalized cosine)
                similarity = sum(q_i * e_i for q_i, e_i in zip(query_embedding, emb))
                results.append({"path": path, "similarity": similarity})
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return [(r["path"], r["similarity"]) for r in results[:limit]]
        return []

    def close(self):
        # print("Mock DB close")
        pass

@pytest.fixture
def mock_vault_setup():
    mock_vault_root = Path("./mock_vault_pytest_backlinker")
    if mock_vault_root.exists():
        shutil.rmtree(mock_vault_root)
    mock_vault_root.mkdir(exist_ok=True)
    
    test_dir = mock_vault_root / "test"
    test_dir.mkdir(exist_ok=True)

    paths = {
        "original": "test/note.md",
        "similar1": "test/similar_note1.md",
        "similar2": "test/similar_note2.md",
        "unrelated": "test/unrelated_note.md",
    }

    contents = {
        "original": "This is the current test note about AI and linking. It could benefit from more connections.",
        "similar1": "Another note talking about AI, which is a very relevant topic.",
        "similar2": "A note about graph databases and linking concepts, also highly relevant.",
        "unrelated": "A recipe for apple pie. Not relevant to AI or linking.",
    }

    (test_dir / "note.md").write_text(contents["original"], encoding="utf-8")
    (test_dir / "similar_note1.md").write_text(contents["similar1"], encoding="utf-8")
    (test_dir / "similar_note2.md").write_text(contents["similar2"], encoding="utf-8")
    (test_dir / "unrelated_note.md").write_text(contents["unrelated"], encoding="utf-8")

    embedding_dim = 3 # Must match the agent's expectation if it has one, or db setup
    db_instance = SimpleMockDb(mock_vault_root, embedding_dim=embedding_dim)
    
    # Add embeddings. Ensure these match what _get_similar_note_paths_INTERNAL expects
    db_instance.add_note_embedding(paths["original"], [0.1, 0.2, 0.3])
    db_instance.add_note_embedding(paths["similar1"], [0.11, 0.22, 0.33]) # Similar
    db_instance.add_note_embedding(paths["similar2"], [0.1, 0.25, 0.35])  # Somewhat similar
    db_instance.add_note_embedding(paths["unrelated"], [0.9, 0.8, 0.7]) # Different


    # Ensure VaultCtx has embedding_dimensions matching the DB and agent's usage
    vault_ctx = VaultCtx(
        root=mock_vault_root,
        db=db_instance,
        kv=None,
        config={},
        embedding_dimensions=embedding_dim 
    )
    
    fixture_data = {
        "root": mock_vault_root,
        "ctx": vault_ctx,
        "paths": paths,
        "contents": contents,
    }
    
    yield fixture_data
    
    # Teardown
    try:
        shutil.rmtree(mock_vault_root)
        # print(f"Cleaned up mock vault: {mock_vault_root}")
    except OSError as e:
        print(f"Error cleaning up mock vault {mock_vault_root}: {e.strerror}")


@pytest.mark.asyncio
async def test_backlinker_adds_relevant_links(mock_vault_setup):
    # Use mock_vault_setup data
    event_path = mock_vault_setup["paths"]["original"]
    vault_ctx = mock_vault_setup["ctx"]
    
    print(f"\n--- Running Pytest Backlinker Test for '{event_path}' ---")
    print(f"Original content of '{event_path}':\n{mock_vault_setup['contents']['original']}")

    await run_backlinker_for_event(
        event_path=event_path,
        event_type="modified", # Event type might influence some agents, generic here
        vault_ctx=vault_ctx
    )
    
    print(f"--- Pytest Backlinker Test for '{event_path}' Finished ---")

    modified_content_path = mock_vault_setup["root"] / event_path
    modified_content = modified_content_path.read_text(encoding="utf-8")
    
    print(f"--- Verifying results for '{event_path}' ---")
    print(f"Content of '{event_path}' after backlinker:\n{modified_content}")

    expected_link_1 = f"[[{mock_vault_setup['paths']['similar1']}]]"
    expected_link_2 = f"[[{mock_vault_setup['paths']['similar2']}]]"
    unexpected_link = f"[[{mock_vault_setup['paths']['unrelated']}]]"

    assert expected_link_1 in modified_content, f"FAIL: Did NOT find link to '{mock_vault_setup['paths']['similar1']}'"
    print(f"PASS: Found link to '{mock_vault_setup['paths']['similar1']}'")
    
    assert expected_link_2 in modified_content, f"FAIL: Did NOT find link to '{mock_vault_setup['paths']['similar2']}'"
    print(f"PASS: Found link to '{mock_vault_setup['paths']['similar2']}'")

    assert unexpected_link not in modified_content, f"FAIL: Incorrectly found link to '{mock_vault_setup['paths']['unrelated']}'"
    print(f"PASS: Correctly did NOT find link to '{mock_vault_setup['paths']['unrelated']}'")

    # Check that the original content was actually changed by adding links
    # This assertion is important because the agent might decide no links are appropriate
    # or the patch application might fail silently in some edge cases.
    # However, if the LLM *correctly* decides no links, this will fail.
    # For this test, we *expect* links to be added.
    assert mock_vault_setup['contents']['original'] != modified_content, "FAIL: Content was not modified by the backlinker agent, but changes were expected."
    print("PASS: Content was modified.")

# --- Tests for _get_similar_note_paths_INTERNAL --- 

@pytest.mark.asyncio
async def test_get_similar_notes_internal_no_db(mock_vault_setup):
    vault_ctx_no_db = mock_vault_setup["ctx"]
    vault_ctx_no_db.db = None # Simulate no DB connection
    current_note_path = mock_vault_setup["paths"]["original"]

    with patch('builtins.print') as mock_print:
        similar_paths = await _get_similar_note_paths_INTERNAL(current_note_path, vault_ctx_no_db)
        assert similar_paths == []
        mock_print.assert_any_call("[_get_similar_note_paths_INTERNAL] Error: Database context not available.")

@pytest.mark.asyncio
async def test_get_similar_notes_internal_embedding_not_found(mock_vault_setup):
    vault_ctx = mock_vault_setup["ctx"]
    current_note_path = "test/non_existent_note_for_embedding.md" # A path not in SimpleMockDb
    
    original_fetchone = vault_ctx.db.fetchone
    def side_effect_fetchone(*args, **kwargs):
        if vault_ctx.db._current_params and vault_ctx.db._current_params[0] == current_note_path:
            return None
        return original_fetchone(*args, **kwargs)
    
    with patch.object(vault_ctx.db, 'fetchone', side_effect=side_effect_fetchone), \
         patch('builtins.print') as mock_print:
        similar_paths = await _get_similar_note_paths_INTERNAL(current_note_path, vault_ctx)
        assert similar_paths == []
        mock_print.assert_any_call(f"[_get_similar_note_paths_INTERNAL] Embedding not found for: {current_note_path}. Cannot find similar notes.")

@pytest.mark.asyncio
async def test_get_similar_notes_internal_db_execute_fails_on_fetch_embedding(mock_vault_setup):
    vault_ctx = mock_vault_setup["ctx"]
    current_note_path = mock_vault_setup["paths"]["original"]

    with patch.object(vault_ctx.db, 'execute', side_effect=Exception("DB error on execute for embedding")) as mock_db_execute, \
         patch('builtins.print') as mock_print:
        similar_paths = await _get_similar_note_paths_INTERNAL(current_note_path, vault_ctx)
        assert similar_paths == []
        mock_print.assert_any_call(f"[_get_similar_note_paths_INTERNAL] Error for {current_note_path}: DB error on execute for embedding")
        mock_db_execute.assert_called_once()
        assert mock_db_execute.call_args[0][0] == "SELECT embedding FROM notes WHERE path = ?"

@pytest.mark.asyncio
async def test_get_similar_notes_internal_db_execute_fails_on_fetch_similar(mock_vault_setup):
    vault_ctx = mock_vault_setup["ctx"]
    current_note_path = mock_vault_setup["paths"]["original"]
    original_embedding = [0.1, 0.2, 0.3]

    mock_db_execute = MagicMock()
    call_count = 0
    def execute_side_effect(query, params=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            if query == "SELECT embedding FROM notes WHERE path = ?" and params[0] == current_note_path:
                mock_conn_for_embedding = MagicMock()
                mock_conn_for_embedding.fetchone.return_value = (original_embedding,)
                return mock_conn_for_embedding
            raise ValueError("Unexpected first DB call")
        elif call_count == 2:
            raise Exception("DB error on execute for similar")
        raise ValueError("Too many DB calls")

    mock_db_execute.side_effect = execute_side_effect
    
    with patch.object(vault_ctx, 'db') as mock_db_attr:
        mock_db_attr.execute = mock_db_execute
        mock_db_attr.embedding_dimensions = vault_ctx.embedding_dimensions
        with patch('builtins.print') as mock_print:
            similar_paths = await _get_similar_note_paths_INTERNAL(current_note_path, vault_ctx)
            assert similar_paths == []
            mock_print.assert_any_call(f"[_get_similar_note_paths_INTERNAL] Error for {current_note_path}: DB error on execute for similar")
            assert mock_db_execute.call_count == 2

# --- Tests for run_backlinker_for_event --- 

@pytest.mark.asyncio
async def test_run_backlinker_event_no_db(mock_vault_setup):
    vault_ctx_no_db = mock_vault_setup["ctx"]
    vault_ctx_no_db.db = None
    event_path = mock_vault_setup["paths"]["original"]
    with patch('builtins.print') as mock_print:
        await run_backlinker_for_event(event_path, "modified", vault_ctx_no_db)
        mock_print.assert_any_call("[Backlinker Runner] Database context not available. Skipping backlinking.")

@pytest.mark.asyncio
async def test_run_backlinker_event_original_note_not_found(mock_vault_setup):
    # This test is PASSING with its minimal strategy (no Path mocks)
    vault_ctx = mock_vault_setup["ctx"]
    event_path_str = "test/non_existent_original.md" # Not in fixture
    with patch('builtins.print') as mock_print:
        await run_backlinker_for_event(event_path_str, "modified", vault_ctx)
        mock_print.assert_any_call(f"[Backlinker Runner] Original note not found: {event_path_str}. Skipping.")

@pytest.mark.asyncio
async def test_run_backlinker_event_no_similar_notes_found(mock_vault_setup, monkeypatch):
    """Test the case where no similar notes are found for the event path."""
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path_str = mock_vault_setup["paths"]["original"]
    expected_original_content = mock_vault_setup["contents"]["original"]
    read_text_called = False
    is_file_called = False
    
    # Create a special mock for Path.resolve() to control path resolution
    original_path_resolve = pathlib.Path.resolve
    
    # Direct monkeypatching for critical path operations
    def mock_read_text(self, encoding=None, **kwargs):
        nonlocal read_text_called
        read_text_called = True
        return expected_original_content
    
    def mock_is_file(self):
        nonlocal is_file_called
        is_file_called = True
        return True
        
    # Apply monkeypatches directly
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)
    
    # Execute test with _get_similar_note_paths_INTERNAL mocked to return empty list
    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', 
              new_callable=AsyncMock, return_value=[]) as mock_get_similar, \
         patch('builtins.print') as mock_print:
        
        # Run the function under test
        await run_backlinker_for_event(event_path_str, "modified", vault_ctx)
        
        # Verify correct function flow
        assert is_file_called, "is_file was not called - path checking didn't execute"
        assert read_text_called, "read_text was not called - expected file content not read"
        mock_get_similar.assert_called_once()
        mock_print.assert_any_call(f"[Backlinker Runner] No similar notes found for {event_path_str}. Nothing to do.")

@pytest.mark.asyncio
async def test_run_backlinker_event_similar_note_file_not_found(mock_vault_setup, monkeypatch):
    """Test the case where a similar note is found but the file doesn't exist."""
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path_str = mock_vault_setup["paths"]["original"]
    similar_path_raw_non_existent = "test/similar_but_missing.md" # Not in fixture
    expected_original_content = mock_vault_setup["contents"]["original"]
    read_text_called = False
    is_file_checks = {}
    
    # Mock path operations
    def mock_read_text(self, encoding=None, **kwargs):
        nonlocal read_text_called
        read_text_called = True
        return expected_original_content
    
    def mock_is_file(self):
        # Track when is_file is called and return appropriate values
        path_str = str(self)
        is_file_checks[path_str] = True  # Record this path was checked
        
        # Return True for the original note, False for the "missing" similar note
        if 'similar_but_missing.md' in path_str:
            return False
        return True
    
    # Apply monkeypatches directly
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)
    
    # Execute test
    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', 
              new_callable=AsyncMock, return_value=[similar_path_raw_non_existent]) as mock_get_similar, \
         patch('builtins.print') as mock_print:
        
        # Run the function under test
        await run_backlinker_for_event(event_path_str, "modified", vault_ctx)
        
        # Verify correct function flow
        assert read_text_called, "read_text was not called - expected file content not read"
        assert any('similar_but_missing' in path for path in is_file_checks.keys()), \
            "No is_file check performed on the missing similar note path"
        mock_get_similar.assert_called_once()
        mock_print.assert_any_call(f"[Backlinker Runner] Similar note file not found: {similar_path_raw_non_existent}. Skipping.")

@pytest.mark.asyncio
async def test_run_backlinker_event_agent_run_exception(mock_vault_setup, monkeypatch):
    """Test the behavior when the backlink agent raises an exception."""
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path_str = mock_vault_setup["paths"]["original"]
    similar_path_str = mock_vault_setup["paths"]["similar1"] # Exists in fixture
    expected_original_content = mock_vault_setup["contents"]["original"]
    expected_similar_content = mock_vault_setup["contents"]["similar1"]
    
    # Track file operations
    paths_read = {}
    paths_checked = {}
    
    # Mock path operations
    def mock_read_text(self, encoding=None, **kwargs):
        path_str = str(self)
        paths_read[path_str] = True  # Record this path was read
        
        # Return appropriate content based on path
        if 'similar_note1.md' in path_str:
            return expected_similar_content
        return expected_original_content
    
    def mock_is_file(self):
        path_str = str(self)
        paths_checked[path_str] = True  # Record this path was checked
        return True  # All files exist in this test
    
    # Apply monkeypatches
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)
    
    # Execute test with agent raising an exception
    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', 
              new_callable=AsyncMock, return_value=[similar_path_str]) as mock_get_similar, \
         patch('obsidian_watchdog.agents.backlinker.backlink_agent.run', 
              new_callable=AsyncMock, side_effect=Exception("Agent Error")) as mock_agent_run, \
         patch('builtins.print') as mock_print:
        
        # Run the function under test
        await run_backlinker_for_event(event_path_str, "modified", vault_ctx)
        
        # Verify correct function flow
        assert any('note.md' in path or event_path_str in path for path in paths_read.keys()), \
            "Original note was not read"
        assert any('similar_note1.md' in path or similar_path_str in path for path in paths_read.keys()), \
            "Similar note was not read"
        mock_get_similar.assert_called_once()
        mock_agent_run.assert_called_once()
        mock_print.assert_any_call(f"[Backlinker Runner] Error running agent for {event_path_str} linking to {similar_path_str}: Agent Error")

@pytest.mark.asyncio
async def test_run_backlinker_event_agent_returns_unexpected_type(mock_vault_setup, comprehensive_path_mocking):
    vault_ctx = mock_vault_setup["ctx"]
    event_path = mock_vault_setup["paths"]["original"]
    similar_path = mock_vault_setup["paths"]["similar1"]
    
    # Set up path mocks
    path_mocks = comprehensive_path_mocking
    path_mocks['is_file'].return_value = True
    path_mocks['exists'].return_value = True
    path_mocks['read_text'].return_value = "Mock content"

    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', new_callable=AsyncMock, return_value=[similar_path]), \
         patch('obsidian_watchdog.agents.backlinker.backlink_agent.run', new_callable=AsyncMock, return_value="not a LinkDecision object") as mock_agent_run, \
         patch('builtins.print') as mock_print:
        await run_backlinker_for_event(event_path, "modified", vault_ctx)
        mock_agent_run.assert_called_once()
        assert any("[Backlinker Runner] Could not extract LinkDecision for" in call.args[0] for call in mock_print.call_args_list)

@pytest.mark.asyncio
async def test_run_backlinker_event_write_patch_fails(mock_vault_setup, monkeypatch):
    """Test the behavior when write_patch fails."""
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path = mock_vault_setup["paths"]["original"]
    similar_path = mock_vault_setup["paths"]["similar1"]
    
    # Create a proper link decision that the agent will return
    mock_link_decision = LinkDecision(
        should_link=True, 
        reason="Test reason", 
        agent_name="BacklinkerAgent"
    )
    
    # Track important function calls
    write_patch_calls = []
    original_print = builtins.print
    print_calls = []
    
    # Mock built-in print to track calls
    def mock_print(*args, **kwargs):
        print_calls.append(args[0] if args else "")
        # Still call original print for debugging
        return original_print(*args, **kwargs)
    
    # Mock path operations
    def mock_read_text(self, encoding=None, **kwargs):
        return "Mock content for testing"
    
    def mock_is_file(self):
        return True  # All files exist in this test

    # Mock write_patch function (reverted to sync, simulates failure)
    def mock_write_patch_sync_fails(*args, **kwargs):
        write_patch_calls.append(args)
        # Simulate failure by returning error message
        return "Error: Patch Failed"

    # Apply all monkeypatches
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)
    
    # Patch the write_patch function where it's used in the backlinker agent
    monkeypatch.setattr("obsidian_watchdog.agents.backlinker.write_patch", mock_write_patch_sync_fails)
    
    # Run the test with the agent returning our link decision
    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', 
              new_callable=AsyncMock, return_value=[similar_path]), \
         patch('obsidian_watchdog.agents.backlinker.backlink_agent.run', 
              new_callable=AsyncMock, return_value=mock_link_decision):
        
        # Execute the function being tested
        await run_backlinker_for_event(event_path, "modified", vault_ctx)
        
        # Verify correct behavior
        assert write_patch_calls, "write_patch was not called"
        assert any(f"Failed to apply cumulative patch to '{event_path}'" in call for call in print_calls), \
            "No failure message for patch application was printed"
        assert not any("Updated last_backlinked_at" in call for call in print_calls), \
            "Database was updated despite patch failure"

@pytest.mark.asyncio
async def test_run_backlinker_event_update_last_backlinked_at_fails(mock_vault_setup, monkeypatch):
    """Test the handling of DB errors when updating the last_backlinked_at field."""
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path = mock_vault_setup["paths"]["original"]
    similar_path = mock_vault_setup["paths"]["similar1"]
    
    # Create a proper link decision that the agent will return
    mock_link_decision = LinkDecision(
        should_link=True, 
        reason="Test reason", 
        agent_name="BacklinkerAgent"
    )
    
    # Track function calls and outputs
    db_execute_calls = []
    print_calls = []
    original_print = builtins.print
    
    # Mock built-in print to track calls
    def mock_print(*args, **kwargs):
        print_calls.append(args[0] if args else "")
        # Still call original print for debugging
        return original_print(*args, **kwargs)
    
    # Mock path operations
    def mock_read_text(self, encoding=None, **kwargs):
        return "Mock content for testing"
    
    def mock_is_file(self):
        return True  # All files exist in this test
    
    # Mock write_patch to return success
    def mock_write_patch(*args, **kwargs):
        return "OK"
    
    # Mock db execute to fail on last_backlinked_at update
    original_db_execute = vault_ctx.db.execute
    def mock_db_execute(query, params=None):
        db_execute_calls.append((query, params))
        if "UPDATE notes SET last_backlinked_at = ?" in query:
            raise Exception("DB error on update last_backlinked_at")
        return original_db_execute(query, params)
    
    # Apply all monkeypatches
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)
    
    # Import and patch the module where write_patch is defined
    import obsidian_watchdog.tools_common
    monkeypatch.setattr(obsidian_watchdog.tools_common, "write_patch", mock_write_patch)
    
    # Run the test
    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL', 
              new_callable=AsyncMock, return_value=[similar_path]), \
         patch('obsidian_watchdog.agents.backlinker.backlink_agent.run', 
              new_callable=AsyncMock, return_value=mock_link_decision), \
         patch.object(vault_ctx.db, 'execute', side_effect=mock_db_execute):
        
        # Execute the function being tested
        await run_backlinker_for_event(event_path, "modified", vault_ctx)
        
        # Verify correct behavior
        assert any(query for query, _ in db_execute_calls if "UPDATE notes SET last_backlinked_at = ?" in query), \
            "Database execute was not called for updating last_backlinked_at"
        assert any(f"Error updating last_backlinked_at for '{event_path}'" in call for call in print_calls), \
            "No error message was printed for the database update failure"

@pytest.mark.asyncio
async def test_run_backlinker_event_general_exception(mock_vault_setup):
    vault_ctx = mock_vault_setup["ctx"]
    event_path = mock_vault_setup["paths"]["original"]
    
    # Different approach - directly cause an exception in Path constructor calls
    def raise_exception(*args, **kwargs):
        raise Exception("General Path Error")
    
    with patch('pathlib.Path', side_effect=raise_exception), \
         patch('builtins.print') as mock_print:
        await run_backlinker_for_event(event_path, "modified", vault_ctx)
        exception_message = f"[Backlinker Runner] General error in run_backlinker_for_event for {event_path}: General Path Error"
        error_messages = [call.args[0] for call in mock_print.call_args_list if "General error" in call.args[0]]
        assert any("General error" in msg for msg in error_messages), f"Did not find error message containing 'General error'. Messages: {error_messages}"

# Need to import LinkDecision for some of the new tests
from obsidian_watchdog.agents.backlinker import LinkDecision
# Need AsyncMock for mocking async functions
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_run_backlinker_event_duplicate_similar_path_skipped(mock_vault_setup, monkeypatch):
    """
    Tests that if _get_similar_note_paths_INTERNAL returns a duplicate path,
    the agent is not called for the duplicate and a skip message is logged.
    """
    # Setup
    vault_ctx = mock_vault_setup["ctx"]
    event_path_str = mock_vault_setup["paths"]["original"]
    similar_path_1 = mock_vault_setup["paths"]["similar1"]
    similar_path_2 = mock_vault_setup["paths"]["similar2"] # A unique path
    # similar_path_1 will be duplicated in the return list

    expected_original_content = mock_vault_setup["contents"]["original"]
    expected_similar1_content = mock_vault_setup["contents"]["similar1"]
    expected_similar2_content = mock_vault_setup["contents"]["similar2"]

    # Mock path operations
    def mock_read_text(self, encoding=None, **kwargs):
        path_str = str(self)
        if event_path_str in path_str:
            return expected_original_content
        elif similar_path_1 in path_str:
            return expected_similar1_content
        elif similar_path_2 in path_str:
            return expected_similar2_content
        return "Default mock content"

    def mock_is_file(self):
        return True # All relevant files exist for this test

    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.Path, "is_file", mock_is_file)
    monkeypatch.setattr(pathlib.WindowsPath, "read_text", mock_read_text)
    monkeypatch.setattr(pathlib.WindowsPath, "is_file", mock_is_file)

    # Agent decision mock
    mock_decision_link = LinkDecision(should_link=True, reason="Test link", agent_name="BacklinkerAgent")

    # Paths to be returned by _get_similar_note_paths_INTERNAL, with a duplicate
    paths_with_duplicate = [similar_path_1, similar_path_2, similar_path_1]

    with patch('obsidian_watchdog.agents.backlinker._get_similar_note_paths_INTERNAL',
               new_callable=AsyncMock, return_value=paths_with_duplicate) as mock_get_similar,\
         patch('obsidian_watchdog.agents.backlinker.backlink_agent.run',\
               new_callable=AsyncMock, return_value=mock_decision_link) as mock_agent_run,\
         patch('builtins.print') as mock_print,\
         patch('obsidian_watchdog.agents.backlinker.write_patch', return_value="OK") as mock_write_patch: # Mock write_patch

        await run_backlinker_for_event(event_path_str, "modified", vault_ctx)

        # Assertions
        mock_get_similar.assert_called_once()
        
        # Agent should be called once for similar_path_1 and once for similar_path_2
        # It should NOT be called for the duplicate similar_path_1
        assert mock_agent_run.call_count == 2, f"Expected agent to be called 2 times, but was called {mock_agent_run.call_count} times."
        
        # Check the arguments of the agent calls
        agent_call_args_list = [call.args[0] for call in mock_agent_run.call_args_list]
        
        # Normalize paths for comparison in prompts
        normalized_similar_path_1 = Path(similar_path_1).as_posix()
        normalized_similar_path_2 = Path(similar_path_2).as_posix()

        prompt_for_similar1_found = any(f"Similar Note ('{normalized_similar_path_1}') Abstract" in args for args in agent_call_args_list)
        prompt_for_similar2_found = any(f"Similar Note ('{normalized_similar_path_2}') Abstract" in args for args in agent_call_args_list)

        assert prompt_for_similar1_found, f"Agent was not called with prompt for {normalized_similar_path_1}"
        assert prompt_for_similar2_found, f"Agent was not called with prompt for {normalized_similar_path_2}"

        # Check for the skip message
        expected_skip_message = f"[Backlinker Runner] Link to '{similar_path_1}' already approved by agent in this run. Skipping duplicate consideration."
        
        # Debug print all mock_print calls if assertion fails
        # for call_args in mock_print.call_args_list:
        # print(f"DEBUG print call: {call_args}")

        assert any(expected_skip_message in call.args[0] for call in mock_print.call_args_list), \
            f"Expected skip message for duplicate path '{similar_path_1}' not found in print logs."
        
        # Ensure write_patch was called, implying links were processed.
        # If agent was called twice, and both returned should_link=True, 
        # and they are different links, then write_patch should be called.
        mock_write_patch.assert_called_once() 