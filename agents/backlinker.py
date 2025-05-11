from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
# from pydantic_ai.tools import Tool # Tool class might be used for more complex tool definitions
from models import Patch
from deps import VaultCtx
from tools_common import read_note, write_patch
from textwrap import dedent
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import Field
from pathlib import Path
from dataclasses import dataclass
import re # For checking existing links

# Configure the model for LM Studio
model = OpenAIModel(
    'qwen3-8b',  # Model name as it appears in LM Studio
    provider=OpenAIProvider(
        base_url='http://192.168.178.20:1234/v1',  # Updated LM Studio IP address and port
        api_key='not-needed'  # LM Studio doesn't require an API key
    )
)

# tools_common.read_note is assumed to be a compatible tool function.
# If it needs RunContext, its signature should be:
# async def read_note(ctx: RunContext[VaultCtx], note_rel_path: str) -> str:
# For now, we rely on its existing definition in tools_common.
backlink_agent = Agent[VaultCtx, Patch](
    model=model,
    deps_type=VaultCtx,
    output_type=Patch,
    system_prompt=dedent(
        """
        You are a note-linking assistant. You will be given the content of an 'Original Note'
        and a 'Similar Note'.
        Your task is to decide if a wikilink to the 'Similar Note' (e.g., [[path/to/similar_note.md]])
        should be intelligently inserted into the 'Original Note'.

        Return a Patch object:
        - 'before': EXACT content of the 'Original Note' as provided.
        - 'after': The 'Original Note' content with the wikilink to the 'Similar Note'
                   intelligently added IF a link is appropriate. If no link is appropriate,
                   'after' should be identical to 'before'.
        - 'comment': Brief explanation of why you did or did not add the link.
        - 'agent': 'BacklinkerAgent'.

        Only add one link per call. Do not link if a link to the 'Similar Note' already exists
        in the 'Original Note'. The link should be the relative path of the similar note.
        Consider the context and relevance before adding a link.
        Place the link where it makes the most sense in the Original Note, or at the end if unsure.
        Ensure the link format is exactly [[path/to/similar_note.md]].
        """
    ),
    instrument=True,  # Enable instrumentation for debugging
)

async def _get_similar_note_paths_INTERNAL(
    current_note_rel_path: str, 
    vault_ctx: VaultCtx, 
    top_k: int = 5
) -> List[str]:
    """
    Internal helper to find similar note paths based on embedding similarity.
    This logic was extracted from the former list_similar_notes tool.
    """
    print(f"[_get_similar_note_paths_INTERNAL] Finding similar notes for {current_note_rel_path}, top_k={top_k}")
    if not vault_ctx.db:
        print("[_get_similar_note_paths_INTERNAL] Error: Database context not available.")
        return []

    try:
        current_note_data = vault_ctx.db.execute(
            "SELECT embedding FROM notes WHERE path = ?", 
            [current_note_rel_path]
        ).fetchone()

        if not current_note_data or not current_note_data[0]:
            print(f"[_get_similar_note_paths_INTERNAL] Embedding not found for: {current_note_rel_path}. Cannot find similar notes.")
            return []
        
        current_embedding = current_note_data[0]
        effective_dimension = vault_ctx.embedding_dimensions

        query = f"""
        SELECT path, array_cosine_similarity(embedding, ?::FLOAT[{effective_dimension}]) AS similarity
        FROM notes
        WHERE path != ?
        ORDER BY similarity DESC
        LIMIT ?;
        """
        similar_notes_rows = vault_ctx.db.execute(
            query, 
            [current_embedding, current_note_rel_path, top_k]
        ).fetchall()
        
        similar_note_paths = [row[0] for row in similar_notes_rows if row[0]]
        print(f"[_get_similar_note_paths_INTERNAL] Found {len(similar_note_paths)} similar notes: {similar_note_paths}")
        return similar_note_paths

    except Exception as e:
        print(f"[_get_similar_note_paths_INTERNAL] Error for {current_note_rel_path}: {e}")
        return []

# Optional: Output Validator for the Patch object
@backlink_agent.output_validator
def ensure_patch_integrity(patch: Patch) -> None:
    """Validates the generated Patch object."""
    if not patch.before:
        raise ValueError("Patch object's 'before' field cannot be empty.")
    if not patch.after:
        raise ValueError("Patch object's 'after' field cannot be empty.")
    if patch.agent != "BacklinkerAgent":
        raise ValueError(f"Patch agent name is incorrect, expected 'BacklinkerAgent', got '{patch.agent}'.")
    
    # If content changed, a comment is good practice.
    if patch.before != patch.after and not patch.comment:
        print(f"[Validator: ensure_patch_integrity] Warning: Patch for agent '{patch.agent}' changed content but has no comment.")
    
    # Check if only one link was added if changes were made
    if patch.before != patch.after:
        # Count wikilinks
        before_links = set(re.findall(r"\[\[(.*?)\]\]", patch.before))
        after_links = set(re.findall(r"\[\[(.*?)\]\]", patch.after))
        newly_added_links = after_links - before_links
        # This validator might be too strict if the agent also reformats or cleans up.
        # For now, we focus on the "single new link" idea.
        # if len(newly_added_links) > 1:
        #     raise ValueError(f"Patch should add at most one new link. Found {len(newly_added_links)} new links: {newly_added_links}")
        # if not newly_added_links and patch.before != patch.after:
        #     # This means content changed but no new link was added. Could be reformatting.
        #     print(f"[Validator: ensure_patch_integrity] Warning: Patch content changed but no new wikilink detected. Check agent's reasoning in comment.")
    
    print(f"[Validator: ensure_patch_integrity] Patch for agent '{patch.agent}' passed basic validation.")

# Example of how you might initialize and use the agent (for testing or direct invocation)
# This part would typically be handled by the router and runner

# FsEvent class is assumed to be defined elsewhere or imported.
# Example:
# from models import FsEvent 

async def run_backlinker_for_event(event_path: str, event_type: str, vault_ctx: VaultCtx):
    print(f"[Backlinker Runner] Orchestrating BacklinkerAgent for event: {event_path} ({event_type})")

    if not vault_ctx.db:
        print("[Backlinker Runner] Database context not available. Skipping backlinking.")
        return

    try:
        # 1. Read the original note's content
        # Use a RunContext to call read_note, as it's a tool expecting it.
        # Or, access file system directly if read_note is simple enough to bypass tool mechanism here.
        # For consistency, let's assume we can get content directly or via a simple helper.
        original_note_full_path = vault_ctx.root / event_path
        if not original_note_full_path.is_file():
            print(f"[Backlinker Runner] Original note not found: {event_path}. Skipping.")
            return
        
        original_content = original_note_full_path.read_text(encoding="utf-8")
        current_content_for_linking = original_content # This will be updated iteratively
        any_changes_made = False

        # 2. Get similar notes
        # No need to pass top_k if we want to process all reasonable similar notes, agent can be selective.
        # Let's stick to a top_k for now to limit processing.
        similar_note_paths = await _get_similar_note_paths_INTERNAL(event_path, vault_ctx, top_k=5)

        if not similar_note_paths:
            print(f"[Backlinker Runner] No similar notes found for {event_path}. Nothing to do.")
            return

        print(f"[Backlinker Runner] Found {len(similar_note_paths)} candidates for {event_path}: {similar_note_paths}")

        # 3. Loop through similar notes and invoke agent for each
        for similar_path in similar_note_paths:
            print(f"[Backlinker Runner] Considering linking {event_path} -> {similar_path}")

            # Check if this link already exists in the most current version of the content
            # This check should ideally use the `current_content_for_linking`
            # Link format: [[path/to/note.md]] or [[note name without extension]]
            # Normalizing the link for checking:
            normalized_similar_path_link = similar_path.replace("\\", "/") # Ensure forward slashes
            if f"[[{normalized_similar_path_link}]]" in current_content_for_linking or \
               f"[[{Path(normalized_similar_path_link).stem}]]" in current_content_for_linking:
                print(f"[Backlinker Runner] Link to '{similar_path}' already exists in current version of '{event_path}'. Skipping.")
                continue
            
            similar_note_full_path = vault_ctx.root / similar_path
            if not similar_note_full_path.is_file():
                print(f"[Backlinker Runner] Similar note file not found: {similar_path}. Skipping.")
                continue
            
            similar_content = similar_note_full_path.read_text(encoding="utf-8")

            user_message = dedent(f"""
            Original Note ('{event_path}') Content:
            ----------------------------------------
            {current_content_for_linking}
            ----------------------------------------

            Similar Note ('{similar_path}') Content:
            ----------------------------------------
            {similar_content}
            ----------------------------------------

            Based on the content of both notes, should a wikilink of the form '[[{similar_path}]]'
            be added to the Original Note content?

            If yes, please return a Patch object where 'before' is the EXACT 'Original Note Content'
            provided above, and 'after' is that content with '[[{similar_path}]]' intelligently inserted.
            If no, 'before' and 'after' should be identical.
            Provide a comment explaining your decision. Agent name: 'BacklinkerAgent'.
            """)
            
            try:
                print(f"[Backlinker Runner] Invoking agent for pair: '{event_path}' and '{similar_path}'...")
                # The 'deps' for the agent run is the vault_ctx
                individual_patch = await backlink_agent.run(user_message, deps=vault_ctx)

                if isinstance(individual_patch, Patch):
                    print(f"[Backlinker Runner] Agent response for {similar_path}: {individual_patch.comment}")
                    if individual_patch.before != individual_patch.after:
                        # Validate that the 'before' from patch matches `current_content_for_linking`
                        if individual_patch.before.strip() != current_content_for_linking.strip():
                            print(f"[Backlinker Runner] CRITICAL MISMATCH: Agent's patch.before does not match current_content_for_linking for {event_path} when processing {similar_path}. This should not happen if prompt is clear.")
                            print("Patch.before:\n", individual_patch.before)
                            print("Current content:\n", current_content_for_linking)
                            # Decide how to handle: skip this patch, or stop? For now, skip.
                            print("[Backlinker Runner] Skipping this erroneous patch.")
                            continue

                        print(f"[Backlinker Runner] Agent suggested changes for link to {similar_path}.")
                        current_content_for_linking = individual_patch.after # Update for next iteration
                        any_changes_made = True
                    else:
                        print(f"[Backlinker Runner] Agent decided no link needed for {similar_path}.")
                else:
                    print(f"[Backlinker Runner] Agent returned an unexpected response type for {similar_path}: {type(individual_patch)}")

            except Exception as agent_e:
                print(f"[Backlinker Runner] Error running agent for {event_path} linking to {similar_path}: {agent_e}")
                import traceback
                traceback.print_exc()
                # Continue to the next similar note

        # 4. After the loop, if any changes were made, apply the final accumulated patch
        if any_changes_made:
            print(f"[Backlinker Runner] Accumulatd changes for {event_path}. Applying final patch.")
            final_patch = Patch(
                before=original_content,  # The very original content
                after=current_content_for_linking, # The content with all links added
                comment=f"Automated backlinking: Iteratively added links based on similarity. Processed {len(similar_note_paths)} candidates.",
                agent="BacklinkerAgent_Orchestrator" # Distinguish from individual agent runs
            )
            
            try:
                dummy_run_context = RunContext(deps=vault_ctx, tool_name="write_patch_orchestrator")
                write_status = write_patch(ctx=dummy_run_context, rel_path=event_path, patch=final_patch)
                print(f"[Backlinker Runner] Final patch application status for {event_path}: {write_status}")
            except Exception as e_patch:
                print(f"[Backlinker Runner] Error applying final accumulated patch to {event_path}: {e_patch}")
        else:
            print(f"[Backlinker Runner] No link changes were made to {event_path} after processing all candidates.")

    except Exception as e:
        print(f"[Backlinker Runner] General error in run_backlinker_for_event for {event_path}: {e}")
        import traceback
        traceback.print_exc()

# If you want to test this file directly:
# if __name__ == "__main__":
#     # This is a very basic mock for testing.
#     # You'd need to set up a dummy VaultCtx and FsEvent.
#     from pathlib import Path # Ensure Path is imported for the mock

#     class MockDb:
#         def execute(self, query, params=None):
#             print(f"Mock DB execute: {query}, {params}")
#             if "SELECT embedding FROM notes" in query and params and params[0] == "test/note.md":
#                 return self # To chain fetchone
#             elif "array_cosine_similarity" in query:
#                 # Simulate finding similar notes
#                 return [("test/similar_note1.md", 0.9), ("test/similar_note2.md", 0.8)]
#             return [] # Default empty result

#         def fetchone(self): # for current_note_data
#             # Simulate finding an embedding for the current note
#             # Ensure dimension matches DB if it were real
#             return ([0.1] * 10,) # Return a tuple with one element (the embedding list)
        
#         def fetchall(self): # for similar_notes_rows
#             # This would be called after an execute that sets up results for fetchall
#             # For simplicity, assume execute returns the list directly if it's a fetchall scenario.
#             # The current mock structure for execute directly returns the list for similarity query.
#             # This fetchall might not be directly hit by list_similar_notes if execute returns full data.
#             # Let's make execute return self for the similarity query too, then fetchall here.
#             # This is getting complex for a mock, the current list_similar_notes expects fetchall() on the cursor.
#             # The execute for similarity should return a cursor-like object.
#             # For now, the existing list_similar_notes mock in main was simpler by directly returning lists.
#             # Let's assume the `execute` in MockDb for similarity query already returns the rows.
#             # So, this fetchall might not be called, or should align with how `execute` is mocked.
#             # The current `list_similar_notes` calls `ctx.deps.db.execute(...).fetchall()`
#             # So the mock execute for similarity should return something that has a fetchall method.
#             # This mock is becoming a bit too involved for a quick test.
#             # Simplification: The provided `main` example had a simpler mock.
#             # Let's stick to a simpler mock for demonstration.
#             return self._last_results if hasattr(self, '_last_results') else []

#         def close(self): print("Mock DB close")

    # tools_common.py would need a mock read_note for this test to run standalone
    # For example:
    # async def mock_read_note(ctx: RunContext[VaultCtx], note_rel_path: str) -> str:
    #     note_full_path = ctx.deps.root / note_rel_path
    #     if note_full_path.exists():
    #         return note_full_path.read_text()
    #     return f"Mock content for {note_rel_path}"
    # Replace read_note in backlink_agent.tools for testing if needed, or ensure tools_common has it.
    # For this example, we assume read_note from tools_common is correctly set up.

#     async def main_test():
#         mock_vault_root = Path("./mock_vault")
#         mock_vault_root.mkdir(exist_ok=True)
#         test_dir = mock_vault_root / "test"
#         test_dir.mkdir(exist_ok=True)
#         (test_dir / "note.md").write_text("This is the current test note about AI and linking.")
#         (test_dir / "similar_note1.md").write_text("Another note talking about AI.")
#         (test_dir / "similar_note2.md").write_text("A note about graph databases and linking concepts.")
#         (test_dir / "unrelated_note.md").write_text("A recipe for apple pie.")

#         # Simplified MockDb for testing list_similar_notes
#         class SimpleMockDb:
#             def __init__(self, vault_root: Path):
#                 self.vault_root = vault_root
#                 self.notes_embeddings = {
#                     "test/note.md": [0.1, 0.2, 0.3],
#                     "test/similar_note1.md": [0.11, 0.22, 0.33], # Similar
#                     "test/similar_note2.md": [0.1, 0.25, 0.35],  # Somewhat similar
#                     "test/unrelated_note.md": [0.9, 0.8, 0.7], # Different
#                 }
#             def execute(self, query: str, params: list = None):
#                 self._current_query = query
#                 self._current_params = params
#                 return self # Return self to allow chaining .fetchone() or .fetchall()

#             def fetchone(self):
#                 if "SELECT embedding FROM notes WHERE path = ?" in self._current_query:
#                     path = self._current_params[0]
#                     embedding = self.notes_embeddings.get(path)
#                     return (embedding,) if embedding else None # Return tuple as DB cursor would
#                 return None

#             def fetchall(self):
#                 if "array_cosine_similarity" in self._current_query:
#                     # This is a very simplified mock of cosine similarity
#                     query_embedding = self._current_params[0]
#                     current_path = self._current_params[1]
#                     limit = self._current_params[2]
                    
#                     results = []
#                     for path, emb in self.notes_embeddings.items():
#                         if path == current_path:
#                             continue
#                         # Pseudo-similarity: sum of products (not normalized)
#                         similarity = sum(q*e for q,e in zip(query_embedding, emb))
#                         results.append({"path": path, "similarity": similarity})
                    
#                     results.sort(key=lambda x: x["similarity"], reverse=True)
#                     return [(r["path"], r["similarity"]) for r in results[:limit]]
#                 return []
#             def close(self): pass

#         mock_db_instance = SimpleMockDb(mock_vault_root)
#         mock_vault_ctx = VaultCtx(root=mock_vault_root, db=mock_db_instance, kv=None, config={})
        
#         # Mocking tools_common.read_note for the test
#         original_read_note = read_note # Save original
        
#         async def mock_read_note_tool(ctx: RunContext[VaultCtx], note_rel_path: str) -> str:
#             print(f"[Mock read_note] Reading {note_rel_path}")
#             full_path = ctx.deps.root / note_rel_path
#             try:
#                 if full_path.is_file():
#                     content = full_path.read_text(encoding="utf-8")
#                     return content
#                 return f"Error: Mock note not found at {note_rel_path}"
#             except Exception as e:
#                 return f"Error reading mock note {note_rel_path}: {str(e)}"

#         # Replace the agent's tool for the duration of the test
#         # This requires backlink_agent.tools to be a mutable list or re-init agent for test
#         # For simplicity, let's assume we can monkeypatch or the test runner handles this.
#         # A more robust way would be to pass tools during agent init for testing.
#         # Global `read_note` is used, so we can just redefine it in this scope for the test.
#         # This is tricky because read_note is imported. For a real test, dependency injection or
#         # a test-specific agent instance would be better.

#         # The tools list on the agent is usually processed at init.
#         # To test with a mock tool, you might need to create a test-specific agent instance.
#         # However, the goal here is to refactor, not perfect the test setup.
#         # Let's assume the global 'read_note' could be temporarily replaced if this was __main__
        
#         # Temporarily replace read_note for testing if it's in the global tools list
#         if backlink_agent.tools_map and 'read_note' in backlink_agent.tools_map:
#              # This is a bit of a hack; proper DI for tools in testing is better
#              # Pydantic AI registers tools by name. We can't easily swap the function pointer
#              # on an existing agent instance this way.
#              # The agent's tools are set at initialization.
#              print("Note: For robust testing with a mock 'read_note', re-initialize 'backlink_agent' with the mock tool.")
#         else: # If read_note is not directly in tools_map, it might be an issue with setup
#              print(f"Warning: 'read_note' tool not found in agent's tool map: {backlink_agent.tools_map.keys()}")


#         print("\\n--- Running Test for 'test/note.md' ---")
#         await run_backlinker_for_event(
#             event_path="test/note.md", 
#             event_type="modified", 
#             vault_ctx=mock_vault_ctx
#         )
    
#     if __name__ == "__main__":
#         # To run this test, ensure FsEvent and VaultCtx are properly defined/imported
#         # and tools_common.read_note is available or mocked.
#         # The mock_read_note_tool needs to be integrated into the agent's toolset for the test.
#         # This can be done by creating a new agent instance for testing with the mocked tool.

#         # For example, to make the test runnable with the mocked read_note:
#         async def mock_read_note_for_main(ctx: RunContext[VaultCtx], note_rel_path: str) -> str:
#             print(f"[Mock read_note for main] Reading {note_rel_path}")
#             full_path = ctx.deps.root / note_rel_path
#             try:
#                 if full_path.is_file():
#                     content = full_path.read_text(encoding="utf-8")
#                     # The agent needs the 'before' content. This tool provides it.
#                     return content 
#                 return f"Error: Mock note not found at {note_rel_path}"
#             except Exception as e:
#                 return f"Error reading mock note {note_rel_path}: {str(e)}"

#         # Store original tools if we were to modify the global agent (not ideal)
#         # original_agent_tools = list(backlink_agent.tools) 

#         # Create a test-specific agent instance for main
#         test_model = OpenAIModel('qwen3-8b', provider=OpenAIProvider(base_url='http://192.168.178.20:1234/v1', api_key='not-needed'))
#         test_backlink_agent = Agent[VaultCtx, Patch](
#             model=test_model,
#             deps_type=VaultCtx,
#             output_type=Patch,
#             system_prompt=backlink_agent.system_prompt_template, # Use existing system prompt
#             tools=[list_similar_notes, mock_read_note_for_main], # Use global list_similar_notes, provide mock_read_note
#             instrument=True
#         )
#         # Re-apply validator if needed (decorators apply to instances)
#         # @test_backlink_agent.output_validator ... (or pass as arg if supported)

#         # Monkey patch the global agent for the test run if needed, or pass test_backlink_agent
#         # to run_backlinker_for_event. Modifying run_backlinker_for_event to accept an agent instance
#         # would be cleaner for testing.
#         # For now, this illustrates the complexity of testing with global agent instances.

#         print("To run the test script: ensure necessary imports (Path, FsEvent, VaultCtx) are resolved,")
#         print("and adapt the 'run_backlinker_for_event' or agent instantiation for testing with mocks.")
#         # asyncio.run(main_test()) # This test setup needs more refinement to work seamlessly.
#         pass # Placeholder to prevent execution of incomplete test setup.

# Mock FsEvent and VaultCtx for the example to be self-contained for linting purposes,
# actual definitions should be imported from their respective modules.
if "FsEvent" not in globals():
    @dataclass
    class FsEvent:
        event_type: str
        path: str
        is_directory: bool = False

if "VaultCtx" not in globals():
    @dataclass
    class VaultCtx:
        root: Path
        db: Any # DuckDB connection or similar
        kv: Any # Key-value store or similar
        config: Dict[str, Any]
        embedding_dimensions: int # Added to mock, as it's used

# The run_backlinker_for_event function needs to be called with an FsEvent-like structure or adapted.
# The current signature is run_backlinker_for_event(event_path: str, event_type: str, vault_ctx: VaultCtx)
# which is fine.

# Cleanup asyncio import if not used elsewhere in the final version of the file.
# It's used by async tool functions and run_backlinker_for_event. 