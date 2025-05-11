import pytest
import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List

# Imports from the project
# Assuming FsEvent and VaultCtx are defined in agents.backlinker or accessible from models
from obsidian_watchdog.agents.backlinker import run_backlinker_for_event, VaultCtx, FsEvent
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