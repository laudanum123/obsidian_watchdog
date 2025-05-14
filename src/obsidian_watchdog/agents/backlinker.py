from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import Usage  # Ensuring this import is present
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field # Added BaseModel, Field
from obsidian_watchdog.models import Patch # EditScriptAction no longer needed here directly
from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.tools_common import write_patch
from textwrap import dedent
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
import re # For checking existing links
import diff_match_patch as dmp_module # For diffing and patching

# New Pydantic model for the LLM's decision
class LinkDecision(BaseModel):
    should_link: bool = Field(description="True if a wikilink to the Similar Note should be added to the Original Note, False otherwise.")
    reason: str = Field(description="A brief explanation for the decision (why the link should or should not be added).")
    agent_name: str = Field(default="BacklinkerAgent", description="Name of the agent making the decision.")

# Configure the model for LM Studio
model = OpenAIModel(
    'qwen3-8b',  # Model name as it appears in LM Studio
    provider=OpenAIProvider(
        base_url='http://192.168.178.20:1234/v1',  # Updated LM Studio IP address and port
        api_key='not-needed'  # LM Studio doesn't require an API key
    )
)

backlink_agent = Agent[VaultCtx, LinkDecision]( # Changed output_type to LinkDecision
    model=model,
    deps_type=VaultCtx,
    output_type=LinkDecision, # Changed output_type to LinkDecision
    system_prompt=dedent(
        """
        You are a note-linking assistant. You will be given the content of an 'Original Note'
        and a 'Similar Note'.
        Your task is to decide if a wikilink to the 'Similar Note' (e.g., [[path/to/similar_note.md]])
        should be added to the 'Original Note'. The link, if added, will be placed at the TOP of the Original Note.

        You MUST return a SINGLE `LinkDecision` object with the following fields:
        - 'should_link': Boolean. True if you recommend adding the link, False otherwise.
        - 'reason': String. A concise explanation for your decision.
        - 'agent_name': String. Must be 'BacklinkerAgent'.

        Guidelines for your decision:
        1. Consider if the 'Similar Note' provides relevant, complementary, or elaborating information
           to the 'Original Note'.
        2. DO NOT recommend linking if a substantially similar link to the 'Similar Note' (checking path and stem)
           already exists in the 'Original Note'.
        3. Base your decision purely on the relevance and non-redundancy of the link.
        
        WHATEVER YOU DO, NEVER CALL final_result MORE THAN ONCE!!!
        Return only the single LinkDecision object.
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

# The ensure_edit_script_action_integrity validator is removed as EditScriptAction is no longer used.
# A new validator for LinkDecision can be added here if needed in the future.

async def run_backlinker_for_event(event_path: str, event_type: str, vault_ctx: VaultCtx):
    print(f"[Backlinker Runner] Orchestrating BacklinkerAgent for event: {event_path} ({event_type})")

    if not vault_ctx.db:
        print("[Backlinker Runner] Database context not available. Skipping backlinking.")
        return

    try:
        original_note_full_path = vault_ctx.root / event_path
        if not original_note_full_path.is_file():
            print(f"[Backlinker Runner] Original note not found: {event_path}. Skipping.")
            return
        
        original_content_on_disk = original_note_full_path.read_text(encoding="utf-8")
        
        similar_note_paths = await _get_similar_note_paths_INTERNAL(event_path, vault_ctx, top_k=5)

        if not similar_note_paths:
            print(f"[Backlinker Runner] No similar notes found for {event_path}. Nothing to do.")
            return

        print(f"[Backlinker Runner] Found {len(similar_note_paths)} candidates for {event_path}: {similar_note_paths}")
        
        links_to_prepend_in_this_run: List[str] = []
        approved_links_details: List[str] = [] # For logging/comment

        for similar_path in similar_note_paths:
            print(f"[Backlinker Runner] Considering linking {event_path} -> {similar_path}")

            normalized_similar_path_link = similar_path.replace("\\\\", "/")
            # Check if link already exists in the original disk content.
            if f"[[{normalized_similar_path_link}]]" in original_content_on_disk or \
               f"[[{Path(normalized_similar_path_link).stem}]]" in original_content_on_disk:
                print(f"[Backlinker Runner] Link to '{similar_path}' already exists in '{event_path}' (on disk). Skipping this candidate.")
                continue
            
            # Also check if we've already decided to add this link in the current run (to avoid duplicates if top_k returns same path multiple times somehow)
            current_link_str_to_check = f"[[{similar_path}]]\\n\\n"
            if current_link_str_to_check in links_to_prepend_in_this_run:
                print(f"[Backlinker Runner] Link to '{similar_path}' already slated for prepending in this run. Skipping duplicate consideration.")
                continue

            similar_note_full_path = vault_ctx.root / similar_path
            if not similar_note_full_path.is_file():
                print(f"[Backlinker Runner] Similar note file not found: {similar_path}. Skipping.")
                continue
            
            similar_content = similar_note_full_path.read_text(encoding="utf-8")

            user_message = dedent(f"""
            Original Note ('{event_path}') Abstract (first 500 chars):
            ----------------------------------------
            {original_content_on_disk[:500]}...
            ----------------------------------------

            Similar Note ('{similar_path}') Abstract (first 500 chars):
            ----------------------------------------
            {similar_content[:500]}...
            ----------------------------------------

            Review the abstracts above. Should a wikilink of the form '[[{similar_path}]]'
            be added to the TOP of the Original Note ('{event_path}')?

            CRITICAL INSTRUCTIONS: You MUST return a SINGLE `LinkDecision` object.
            NEVER CALL final_result MORE THAN ONCE!!!
            The object should have:
            - 'should_link': true or false.
            - 'reason': Your concise explanation.
            - 'agent_name': 'BacklinkerAgent'.
            """)
            
            try:
                print(f"[Backlinker Runner] Invoking agent for pair: '{event_path}' and '{similar_path}'...")
                agent_run_result = await backlink_agent.run(user_message, deps=vault_ctx)

                link_decision_output: Optional[LinkDecision] = None
                if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, LinkDecision):
                    link_decision_output = agent_run_result.output
                elif isinstance(agent_run_result, LinkDecision):
                    link_decision_output = agent_run_result
                
                if link_decision_output:
                    decision = link_decision_output
                    print(f"[Backlinker Runner] Agent decision for '{similar_path}': should_link={decision.should_link}. Reason: {decision.reason}")
                    
                    if decision.should_link:
                        link_to_add = f"[[{similar_path}]]\\n\\n"
                        links_to_prepend_in_this_run.append(link_to_add)
                        approved_links_details.append(f"[[{similar_path}]] (Reason: {decision.reason})")
                        print(f"[Backlinker Runner] Agent approved link to '{similar_path}'. Queued for prepending.")
                else:
                    log_message = f"[Backlinker Runner] Could not extract LinkDecision for {similar_path}.\n"
                    log_message += f"  Type of agent_run_result: {type(agent_run_result)}\n"
                    if hasattr(agent_run_result, 'output'):
                        log_message += f"  Type of agent_run_result.output: {type(agent_run_result.output)}\n"
                        log_message += f"  Content of agent_run_result.output (first 500 chars): {str(agent_run_result.output)[:500]}...\n"
                    else:
                        log_message += "  agent_run_result has no 'output' attribute.\n"
                    log_message += f"  Full agent_run_result (first 500 chars): {str(agent_run_result)[:500]}..."
                    print(log_message)

            except Exception as agent_e:
                print(f"[Backlinker Runner] Error running agent for {event_path} linking to {similar_path}: {agent_e}")
                import traceback
                traceback.print_exc()
        
        # After iterating through all similar notes, apply the accumulated patch if any links were approved.
        if links_to_prepend_in_this_run:
            all_new_links_string = "".join(links_to_prepend_in_this_run)
            modified_content = all_new_links_string + original_content_on_disk
            
            patch_comment = f"Agent 'BacklinkerAgent' decided to add the following links to the top: {'; '.join(approved_links_details)}"
            final_patch = Patch(
                before=original_content_on_disk,
                after=modified_content,
                comment=patch_comment,
                agent="BacklinkerAgent_Orchestrator" # Using a distinct agent name for the cumulative patch
            )
            
            print(f"[Backlinker Runner] Approved {len(links_to_prepend_in_this_run)} link(s) for '{event_path}'. Preparing to write single patch.")
            dummy_run_context = RunContext(
                deps=vault_ctx, 
                tool_name="write_patch_orchestrator",
                model="dummy_model_for_tool_call",
                usage=Usage(),
                prompt="dummy_prompt_for_tool_call"
            )
            write_status = write_patch(ctx=dummy_run_context, rel_path=event_path, patch=final_patch)
            print(f"[Backlinker Runner] Cumulative patch application status for '{event_path}': {write_status}")
            if write_status.lower() == "ok":
                print(f"[Backlinker Runner] Successfully applied cumulative patch to '{event_path}'.")
                # Update last_backlinked_at for all chunks of this note
                try:
                    from datetime import datetime, timezone
                    now_utc = datetime.now(timezone.utc)
                    vault_ctx.db.execute(
                        "UPDATE notes SET last_backlinked_at = ? WHERE path = ?",
                        [now_utc, event_path]
                    )
                    print(f"[Backlinker Runner] Updated last_backlinked_at for '{event_path}' in DB.")
                except Exception as e:
                    print(f"[Backlinker Runner] Error updating last_backlinked_at for '{event_path}': {e}")
            else:
                print(f"[Backlinker Runner] Failed to apply cumulative patch to '{event_path}'.")
        else:
            print(f"[Backlinker Runner] No new links were approved for '{event_path}' during this run.")

    except Exception as e:
        print(f"[Backlinker Runner] General error in run_backlinker_for_event for {event_path}: {e}")
        import traceback
        traceback.print_exc()

# Mock FsEvent and VaultCtx for the example to be self-contained for linting purposes,
# actual definitions should be imported from their respective modules.
# Ensure these are defined if not imported from elsewhere when running standalone.
if "FsEvent" not in globals():
    @dataclass
    class FsEvent:
        event_type: str
        path: str
        is_directory: bool = False

if "VaultCtx" not in globals(): # VaultCtx is defined above if running this file
    pass

# Cleanup asyncio import if not used elsewhere in the final version of the file.
# It's used by async tool functions and run_backlinker_for_event. 

# --- Batch backlinker for all notes needing processing ---
async def run_backlinker_for_all_notes(vault_ctx: VaultCtx):
    """
    Runs the backlinker for all notes in the vault that have not been processed
    since their last modification (last_backlinked_at is NULL or modified_at > last_backlinked_at).
    """
    print("[Batch Backlinker] Starting batch backlinking for all notes needing processing...")
    if not vault_ctx.db:
        print("[Batch Backlinker] No DB connection in vault_ctx. Aborting.")
        return
    try:
        # Get all unique note paths that need backlinking
        rows = vault_ctx.db.execute(
            """
            SELECT DISTINCT path FROM notes
            WHERE last_backlinked_at IS NULL OR modified_at > last_backlinked_at
            """
        ).fetchall()
        note_paths = [row[0] for row in rows]
        print(f"[Batch Backlinker] Found {len(note_paths)} notes to process.")
        for idx, path in enumerate(note_paths, 1):
            print(f"[Batch Backlinker] ({idx}/{len(note_paths)}) Processing: {path}")
            await run_backlinker_for_event(event_path=path, event_type="batch", vault_ctx=vault_ctx)
        print("[Batch Backlinker] Finished batch backlinking for all notes.")
    except Exception as e:
        print(f"[Batch Backlinker] Error during batch backlinking: {e}") 