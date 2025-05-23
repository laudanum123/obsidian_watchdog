from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import Usage  # Ensuring this import is present
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field # Added BaseModel, Field
from obsidian_watchdog.models import Patch # EditScriptAction no longer needed here directly
from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.tools_common import write_patch
from textwrap import dedent
from typing import List, Optional, Tuple, Set # Added Tuple, Set
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
        
        print("BACKLINKER DEBUG: type(original_note_full_path) =", type(original_note_full_path))
        print("BACKLINKER DEBUG: dir(original_note_full_path) =", dir(original_note_full_path))
        original_content_on_disk = original_note_full_path.read_text(encoding="utf-8")
        
        similar_note_paths = await _get_similar_note_paths_INTERNAL(event_path, vault_ctx, top_k=5)

        if not similar_note_paths:
            print(f"[Backlinker Runner] No similar notes found for {event_path}. Nothing to do.")
            return

        print(f"[Backlinker Runner] Found {len(similar_note_paths)} candidates for {event_path}: {similar_note_paths}")
        
        # Store (similar_path_raw, reason_from_agent) for agent-approved links
        agent_approved_actions: List[Tuple[str, str]] = [] 

        for similar_path_raw in similar_note_paths: # Renamed for clarity
            print(f"[Backlinker Runner] Considering linking {event_path} -> {similar_path_raw}")

            # Normalize path for checks and link creation (use forward slashes)
            normalized_similar_path = Path(similar_path_raw).as_posix()

            # Check 1: Link already exists ANYWHERE in the note (wikilink or stem)
            if f"[[{normalized_similar_path}]]" in original_content_on_disk or \
               f"[[{Path(normalized_similar_path).stem}]]" in original_content_on_disk:
                print(f"[Backlinker Runner] Link to '{normalized_similar_path}' (or its stem) already exists anywhere in '{event_path}'. Skipping this candidate.")
                continue
            
            # Check 2: If this similar_path_raw has already been approved by the agent in *this current run*
            if any(approved_path == similar_path_raw for approved_path, _ in agent_approved_actions):
                print(f"[Backlinker Runner] Link to '{similar_path_raw}' already approved by agent in this run. Skipping duplicate consideration.")
                continue

            similar_note_full_path = vault_ctx.root / similar_path_raw # Use raw path for file system access
            if not similar_note_full_path.is_file():
                print(f"[Backlinker Runner] Similar note file not found: {similar_path_raw}. Skipping.")
                continue
            
            similar_content = similar_note_full_path.read_text(encoding="utf-8")

            user_message = dedent(f"""
            Original Note ('{event_path}') Abstract (first 500 chars):
            ----------------------------------------
            {original_content_on_disk[:500]}...
            ----------------------------------------

            Similar Note ('{normalized_similar_path}') Abstract (first 500 chars):
            ----------------------------------------
            {similar_content[:500]}...
            ----------------------------------------

            Review the abstracts above. Should a wikilink of the form '[[{normalized_similar_path}]]'
            be added to the Original Note ('{event_path}')? Your decision will determine if it's added under a "## Backlinks" section.

            CRITICAL INSTRUCTIONS: You MUST return a SINGLE `LinkDecision` object.
            NEVER CALL final_result MORE THAN ONCE!!!
            The object should have:
            - 'should_link': true or false.
            - 'reason': Your concise explanation for why it's relevant or not. This reason will be shown next to the link.
            - 'agent_name': 'BacklinkerAgent'.
            """)
            
            try:
                print(f"[Backlinker Runner] Invoking agent for pair: '{event_path}' and '{similar_path_raw}'...")
                agent_run_result = await backlink_agent.run(user_message, deps=vault_ctx)

                link_decision_output: Optional[LinkDecision] = None
                if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, LinkDecision):
                    link_decision_output = agent_run_result.output
                elif isinstance(agent_run_result, LinkDecision):
                    link_decision_output = agent_run_result
                
                if link_decision_output:
                    decision = link_decision_output
                    print(f"[Backlinker Runner] Agent decision for '{similar_path_raw}': should_link={decision.should_link}. Reason: {decision.reason}")
                    
                    if decision.should_link:
                        agent_approved_actions.append((similar_path_raw, decision.reason))
                        print(f"[Backlinker Runner] Agent approved link to '{similar_path_raw}'. Reason: {decision.reason}. Queued for '## Backlinks' section.")
                else:
                    log_message = f"[Backlinker Runner] Could not extract LinkDecision for {similar_path_raw}.\n"
                    log_message += f"  Type of agent_run_result: {type(agent_run_result)}\n"
                    if hasattr(agent_run_result, 'output'):
                        log_message += f"  Type of agent_run_result.output: {type(agent_run_result.output)}\n"
                        log_message += f"  Content of agent_run_result.output (first 500 chars): {str(agent_run_result.output)[:500]}...\n"
                    else:
                        log_message += "  agent_run_result has no 'output' attribute.\n"
                    log_message += f"  Full agent_run_result (first 500 chars): {str(agent_run_result)[:500]}..."
                    print(log_message)

            except Exception as agent_e:
                print(f"[Backlinker Runner] Error running agent for {event_path} linking to {similar_path_raw}: {agent_e}")
                import traceback
                traceback.print_exc()
        
        if not agent_approved_actions:
            print(f"[Backlinker Runner] No new links were approved by the agent for '{event_path}' or all candidates were already handled.")
            return

        new_bullets_to_render: List[str] = []
        patch_comment_details: List[str] = []
        added_to_bullets_in_this_run_normalized: Set[str] = set() 

        backlinks_heading_text = "## Backlinks"
        backlinks_section_regex = re.compile(
            rf"(^{re.escape(backlinks_heading_text)}\n)(.*?)($|\n## )", 
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        existing_bullet_link_regex = re.compile(r"-\s*\[\[([^\]]+)\]\]")

        current_content_for_processing = original_content_on_disk
        section_match = backlinks_section_regex.search(current_content_for_processing)

        existing_links_in_section_normalized: Set[str] = set()
        if section_match:
            existing_bullets_area_text = section_match.group(2)
            for line in existing_bullets_area_text.split('\n'):
                bullet_link_match = existing_bullet_link_regex.search(line)
                if bullet_link_match:
                    existing_links_in_section_normalized.add(Path(bullet_link_match.group(1)).as_posix())
        
        for raw_path, reason in agent_approved_actions:
            normalized_path = Path(raw_path).as_posix()
            if normalized_path not in existing_links_in_section_normalized and \
               normalized_path not in added_to_bullets_in_this_run_normalized:
                
                bullet_item_text = f"- [[{normalized_path}]] - {reason}"
                new_bullets_to_render.append(bullet_item_text)
                patch_comment_details.append(f"[[{normalized_path}]] ({reason})")
                added_to_bullets_in_this_run_normalized.add(normalized_path)
            else:
                if normalized_path in existing_links_in_section_normalized:
                    print(f"[Backlinker Runner] Link to '{normalized_path}' already present in '## Backlinks' section of '{event_path}'. Not adding duplicate bullet.")
                else: # in added_to_bullets_in_this_run_normalized
                    print(f"[Backlinker Runner] Link to '{normalized_path}' already queued for '## Backlinks' section in this run. Not adding duplicate bullet.")

        if not new_bullets_to_render:
            print(f"[Backlinker Runner] All agent-approved links for '{event_path}' are already present in the '## Backlinks' section or were duplicates in this run. No changes to make to section.")
            return

        new_bullets_block_text = "\n".join(new_bullets_to_render)
        final_modified_content: str

        if section_match:
            content_before_bullets_area = current_content_for_processing[:section_match.start(2)]
            content_after_bullets_area = current_content_for_processing[section_match.end(2):]
            
            existing_bullets_text_stripped = section_match.group(2).strip()
            updated_bullets_area_text: str
            if existing_bullets_text_stripped:
                updated_bullets_area_text = existing_bullets_text_stripped + "\n" + new_bullets_block_text
            else:
                updated_bullets_area_text = new_bullets_block_text
            
            if updated_bullets_area_text: # Ensure single trailing newline if content exists
                 updated_bullets_area_text = updated_bullets_area_text.rstrip() + "\n"

            final_modified_content = content_before_bullets_area + updated_bullets_area_text + content_after_bullets_area
        else:
            heading_to_add_str = f"\n\n{backlinks_heading_text}\n"
            if not current_content_for_processing.strip():
                heading_to_add_str = f"{backlinks_heading_text}\n"
            elif not current_content_for_processing.endswith("\n"):
                heading_to_add_str = f"\n\n{backlinks_heading_text}\n"
            elif current_content_for_processing.endswith("\n\n"):
                 heading_to_add_str = f"{backlinks_heading_text}\n"
            else: 
                 heading_to_add_str = f"\n{backlinks_heading_text}\n"

            final_modified_content = current_content_for_processing.rstrip() + heading_to_add_str + new_bullets_block_text
            if not final_modified_content.endswith("\n"): # Ensure trailing newline for new section at EOF
                final_modified_content += "\n"


        if final_modified_content.strip() == original_content_on_disk.strip():
            print(f"[Backlinker Runner] No effective change to content for '{event_path}' after processing. Skipping patch.")
            return

        patch_comment_for_tool = f"Agent 'BacklinkerAgent' updated '## Backlinks' section in '{event_path}' with: {'; '.join(patch_comment_details)}"
        final_patch = Patch(
            action="MODIFY",
            target_path=event_path,
            content=final_modified_content,
            event_path=event_path,
            before=original_content_on_disk,
            after=final_modified_content,
            comment=patch_comment_for_tool,
            agent="BacklinkerAgent_Orchestrator"
        )
        
        print(f"[Backlinker Runner] {len(new_bullets_to_render)} new unique link(s) identified for '## Backlinks' section in '{event_path}'. Preparing to write single patch.")
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