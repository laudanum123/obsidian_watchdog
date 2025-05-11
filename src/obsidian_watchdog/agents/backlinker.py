from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import Usage  # Ensuring this import is present
from pydantic_ai.providers.openai import OpenAIProvider
# from pydantic_ai.tools import Tool # Tool class might be used for more complex tool definitions
from obsidian_watchdog.models import Patch, EditScriptAction
from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.tools_common import write_patch
from textwrap import dedent
from typing import List
from pathlib import Path
from dataclasses import dataclass
import re # For checking existing links
import diff_match_patch as dmp_module # For diffing and patching

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
backlink_agent = Agent[VaultCtx, EditScriptAction](
    model=model,
    deps_type=VaultCtx,
    output_type=EditScriptAction,
    system_prompt=dedent(
        """
        You are a note-linking assistant. You will be given the content of an 'Original Note'
        and a 'Similar Note'.
        Your task is to decide if a wikilink to the 'Similar Note' (e.g., [[path/to/similar_note.md]])
        should be intelligently inserted into the 'Original Note'.

        Return an EditScriptAction object:
        - 'script': This field should contain the NEW version of the specific paragraph or minimal set of lines
                    from the 'Original Note' (provided in the user message), WITH the wikilink added.
                    If no link is appropriate, this should be empty.
        - 'original_context_for_script': This field should contain the EXACT, VERBATIM lines from the
                                         'Original Note' that your 'script' field is intended to replace.
                                         This helps locate where to apply the change.
                                         If no link is appropriate, this should be empty.
        - 'comment': Brief explanation of why you did or did not add the link.
        - 'agent_name': 'BacklinkerAgent'.
        - 'no_change_needed': Set to True if no link is appropriate and no changes are made, otherwise False.

        Only add one link per call. Do not link if a link to the 'Similar Note' already exists
        in the 'Original Note'. The link should be the relative path of the similar note.
        Consider the context and relevance before adding a link.
        Place the link where it makes the most sense in the Original Note.
        Ensure the link format is exactly [[path/to/similar_note.md]].
        If adding a link, ensure 'script' contains the modified text segment and 'original_context_for_script'
        contains the corresponding original segment.
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
def ensure_edit_script_action_integrity(action: EditScriptAction) -> EditScriptAction:
    """Validates the generated EditScriptAction object."""
    if action.agent_name != "BacklinkerAgent":
        raise ValueError(f"Patch agent name is incorrect, expected 'BacklinkerAgent', got '{action.agent_name}'.")

    if action.no_change_needed:
        if action.script or action.original_context_for_script:
            # Allow comment even if no change is needed
            if action.script:
                 print(f"[Validator: ensure_edit_script_action_integrity] Warning: no_change_needed is true but script is not empty for agent '{action.agent_name}'.")
            if action.original_context_for_script:
                 print(f"[Validator: ensure_edit_script_action_integrity] Warning: no_change_needed is true but original_context_for_script is not empty for agent '{action.agent_name}'.")

    else: # Changes are intended
        if not action.script: # New content block
            raise ValueError("If changes are made (no_change_needed is false), 'script' (new content block) cannot be empty.")
        if not action.original_context_for_script: # Old content block
            raise ValueError("If changes are made (no_change_needed is false), 'original_context_for_script' (old content block) cannot be empty.")
        if not action.comment:
            print(f"[Validator: ensure_edit_script_action_integrity] Warning: EditScriptAction for agent '{action.agent_name}' has changes but no comment.")
        if action.script == action.original_context_for_script:
             print(f"[Validator: ensure_edit_script_action_integrity] Warning: script and original_context_for_script are identical for agent '{action.agent_name}'. Agent should set no_change_needed=True or provide a reason in comment.")

    if not action.no_change_needed and action.script and action.original_context_for_script:
        original_links = set(re.findall(r"\[\[(.*?)\]\]", action.original_context_for_script))
        new_links = set(re.findall(r"\[\[(.*?)\]\]", action.script))
        added_links = new_links - original_links
        if len(added_links) > 1:
            print(f"[Validator: ensure_edit_script_action_integrity] Warning: Agent '{action.agent_name}' script suggests adding {len(added_links)} new links: {added_links}. Expected one.")

    print(f"[Validator: ensure_edit_script_action_integrity] EditScriptAction for agent '{action.agent_name}' passed validation.")
    return action

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
        dmp = dmp_module.diff_match_patch() # Initialize diff-match-patch

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

            If yes, please return an EditScriptAction object:
            - 'script': The NEW version of the paragraph or minimal lines from 'Original Note Content' (above)
                        WITH '[[{similar_path}]]' intelligently inserted.
            - 'original_context_for_script': The EXACT, VERBATIM lines from 'Original Note Content' (above)
                                             that your 'script' field is intended to replace.
            - 'comment': Your explanation.
            - 'agent_name': 'BacklinkerAgent'.
            - 'no_change_needed': True if no link should be added, otherwise False.
            Ensure 'script' and 'original_context_for_script' are empty if 'no_change_needed' is True.
            If 'no_change_needed' is False, ensure both 'script' and 'original_context_for_script' are populated.
            """)
            
            try:
                print(f"[Backlinker Runner] Invoking agent for pair: '{event_path}' and '{similar_path}'...")
                # The 'deps' for the agent run is the vault_ctx
                agent_run_result = await backlink_agent.run(user_message, deps=vault_ctx)

                # Attempt to extract the EditScriptAction from the agent_run_result
                actual_edit_action = None
                if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, EditScriptAction):
                    actual_edit_action = agent_run_result.output
                elif isinstance(agent_run_result, EditScriptAction): # Should ideally be this
                    actual_edit_action = agent_run_result
                
                if actual_edit_action:
                    edit_action = actual_edit_action # Use the extracted/correct action
                    print(f"[Backlinker Runner] Agent response for {similar_path}: {edit_action.comment}")
                    
                    if not edit_action.no_change_needed and edit_action.script and edit_action.original_context_for_script:
                        if edit_action.script == edit_action.original_context_for_script:
                            print(f"[Backlinker Runner] Agent suggested no effective change for {similar_path} (script and original_context match). Comment: {edit_action.comment}")
                        else:
                            patches = dmp.patch_make(edit_action.original_context_for_script, edit_action.script)
                            new_content, success_array = dmp.patch_apply(patches, current_content_for_linking)

                            if all(success_array):
                                if current_content_for_linking != new_content:
                                    print(f"[Backlinker Runner] Agent suggested changes for link to {similar_path}. Applying.")
                                    current_content_for_linking = new_content
                                    any_changes_made = True
                                else:
                                    print(f"[Backlinker Runner] Patch applied but resulted in no change to content for {similar_path}.")
                            else:
                                print(f"[Backlinker Runner] CRITICAL: Patch application failed for {event_path} linking to {similar_path}. Success flags: {success_array}")
                                print(f"  Original context from agent: '''{edit_action.original_context_for_script}'''")
                                print(f"  Script from agent: '''{edit_action.script}'''")
                                # Decide how to handle: skip this patch, or stop? For now, skip.
                                print("[Backlinker Runner] Skipping this failed patch.")
                                continue # Skip to next similar_path
                    elif edit_action.no_change_needed:
                        print(f"[Backlinker Runner] Agent decided no link needed for {similar_path}. Comment: {edit_action.comment}")
                    else: # no_change_needed is False, but script or original_context might be missing (validator should catch this)
                        print(f"[Backlinker Runner] Agent indicated changes for {similar_path} but script/original_context might be incomplete. no_change_needed={edit_action.no_change_needed}. Comment: {edit_action.comment}")
                        print(f"  Script: '''{edit_action.script}'''")
                        print(f"  Original Context: '''{edit_action.original_context_for_script}'''")
                else: # This block means actual_edit_action was None
                    log_message = f"[Backlinker Runner] Could not extract EditScriptAction for {similar_path}.\n"
                    log_message += f"  Type of agent_run_result: {type(agent_run_result)}\n"
                    if hasattr(agent_run_result, 'output'):
                        log_message += f"  Type of agent_run_result.output: {type(agent_run_result.output)}\n"
                        log_message += f"  Content of agent_run_result.output (first 500 chars): {str(agent_run_result.output)[:500]}...\n"
                        # Also log if output is a list and what its elements are
                        if isinstance(agent_run_result.output, list) and len(agent_run_result.output) > 0:
                            log_message += f"  Type of agent_run_result.output[0]: {type(agent_run_result.output[0])}\n"
                            log_message += f"  Content of agent_run_result.output[0] (first 300 chars): {str(agent_run_result.output[0])[:300]}...\n"
                    else:
                        log_message += "  agent_run_result has no 'output' attribute.\n"
                    log_message += f"  Full agent_run_result (first 500 chars): {str(agent_run_result)[:500]}..."
                    print(log_message)

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
                dummy_run_context = RunContext(
                    deps=vault_ctx, 
                    tool_name="write_patch_orchestrator",
                    model="dummy_model_for_tool_call",  # Placeholder
                    usage=Usage(),  # Placeholder
                    prompt="dummy_prompt_for_tool_call"  # Placeholder
                )
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
    # Check if VaultCtx is already defined to avoid redefinition issues
    # This might require VaultCtx to be defined earlier in the file or imported
    # For this specific edit, we assume VaultCtx is defined earlier or imported
    pass

# Cleanup asyncio import if not used elsewhere in the final version of the file.
# It's used by async tool functions and run_backlinker_for_event. 