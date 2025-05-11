from datetime import datetime
from pydantic_ai import RunContext
from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.models import Patch

# Note: pydantic_ai.RunContext might need to be imported differently 
# depending on the version or specific structure of the pydantic_ai library.
# If pydantic_ai.RunContext is not found, it might be pydantic_ai.Agent.RunContext or similar.

def read_note(ctx: RunContext[VaultCtx], rel_path: str) -> str:
    """Return raw Markdown of a note inside the vault."""
    full_path = ctx.deps.root / rel_path
    
    return full_path.read_text(encoding="utf-8")

def write_patch(ctx: RunContext[VaultCtx], rel_path: str, patch: Patch) -> str:
    """Apply a validated Patch object to the note and return 'ok'."""
    fp = ctx.deps.root / rel_path
    # Ensure the 'before' content matches the current file content before patching
    # This is a safety check, though more robust diff/patch mechanisms are better for production
    current_content = fp.read_text(encoding="utf-8")
    
    # Loop prevention: If the proposed 'after' content is identical to current disk content, do nothing.
    if current_content == patch.after:
        print(f"[write_patch] Content for {rel_path} is already identical to patch.after. No changes made.")
        return "no_change_identical_content"

    if current_content != patch.before:
        print(f"Warning: Content of {rel_path} has changed since patch was generated (patch.before mismatch). Proceeding with write.")
        # Depending on strategy, you might raise an error, try to merge/reject, or re-generate patch.
        # For now, we log and proceed.
    
    fp.write_text(patch.after, encoding="utf-8")
    
    # Log the applied patch to the key-value store (TinyDB)
    # Ensure ctx.deps.kv is initialized and supports an 'insert' like method
    if hasattr(ctx.deps.kv, 'insert'):
        try:
            # The original blueprint uses patch.dict(), which is for Pydantic v1.
            # For Pydantic v2, it's model_dump().
            patch_data_to_log = patch.model_dump() if hasattr(patch, 'model_dump') else patch.dict()
            ctx.deps.kv.insert({
                "applied_patch_to_file": rel_path,
                "patch_details": patch_data_to_log,
                "agent_responsible": patch.agent,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            print(f"Error logging patch to KV store: {e}")
            # Decide if this error should prevent returning "ok"
    
    # If write was successful, record it to potentially ignore self-generated events
    ctx.deps.add_agent_modified_path(rel_path)
    return "ok"

# Example of a tool that might be used by an agent for more complex reasoning 