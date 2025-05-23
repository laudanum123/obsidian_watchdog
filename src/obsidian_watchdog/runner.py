import asyncio
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from typing import Dict, Tuple, Deque, Optional
from collections import deque
import hashlib

from obsidian_watchdog.models import FsEvent, Patch
from pydantic_ai import Agent
from obsidian_watchdog.deps import VaultCtx
from obsidian_watchdog.router import get_router

# Configuration for the event batcher
EVENT_BATCH_WINDOW_SECONDS = 3.0
MAX_QUEUE_WAIT_SECONDS = 1.0
PURGE_OLD_EVENTS_SECONDS = 10.0 # Time to wait before flushing an event if no new ones arrive
HANDLER_DEBOUNCE_SECONDS = 1.5 # Debounce window for ChangeHandler

class ChangeHandler(FileSystemEventHandler):
    """Handles file system events and puts them onto an asyncio queue."""
    def __init__(self, queue: asyncio.Queue, vault_root: Path, vault_ctx: VaultCtx):
        super().__init__()
        self.queue = queue
        self.vault_root = vault_root
        self.vault_ctx = vault_ctx
        # Define ignore patterns relative to the vault root
        self.ignore_patterns = [".git/", ".obsidian/", "ai_logs/", "node_modules/", ".DS_Store"]
        self.ignore_extensions = [".tmp", "~", ".swp", ".swx", ".crswap"] # Temp/swap file extensions
        print(f"[ChangeHandler] Initialized. Watching: {self.vault_root}")
        print(f"[ChangeHandler] Ignoring patterns starting with: {self.ignore_patterns}")
        print(f"[ChangeHandler] Ignoring extensions: {self.ignore_extensions}")
        self._last_event_times: Dict[Path, float] = {} # For debouncing

    def _should_ignore(self, event_path: Path) -> bool:
        # Convert absolute event_path to relative path string for was_recently_modified_by_agent
        try:
            rel_path_str_for_check = event_path.relative_to(self.vault_root).as_posix()
            if self.vault_ctx.was_recently_modified_by_agent(rel_path_str_for_check):
                print(f"[ChangeHandler] Ignoring event for {rel_path_str_for_check} as it was recently modified by an agent.")
                return True
        except ValueError: # Should not happen if event_path is within vault_root, but as a safeguard
            pass # If it's outside, other ignore logic will catch it

        path_str = str(event_path)
        if any(path_str.endswith(ext) for ext in self.ignore_extensions):
            # print(f"[ChangeHandler] Ignoring file by extension: {path_str}")
            return True
        try:
            rel_path_str = event_path.relative_to(self.vault_root).as_posix()
            if any(rel_path_str.startswith(pattern) for pattern in self.ignore_patterns):
                # print(f"[ChangeHandler] Ignoring path by pattern: {rel_path_str}")
                return True
        except ValueError: 
            return True
        return False

    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return

        event_abs_path = Path(event.src_path).resolve()

        if self._should_ignore(event_abs_path):
            return

        current_event_type = event.event_type
        mapped_event_type: Optional[str] = None

        if current_event_type == 'moved':
            mapped_event_type = "modified"
            # print(f"[ChangeHandler] Raw event 'moved', mapping to '{mapped_event_type}' for path: {event_abs_path}")
        elif current_event_type in ["created", "modified", "deleted"]:
            mapped_event_type = current_event_type
        else:
            # This is an event type Watchdog might report but we don't specifically handle (e.g., 'opened', 'closed')
            # These events should not reset or participate in the debounce timer for create/modify/delete.
            print(f"[ChangeHandler] Ignoring unhandled watchdog event_type '{current_event_type}' for {event_abs_path}")
            return

        # At this point, mapped_event_type is "created", "modified", or "deleted".
        # Now, apply debounce logic for these relevant types.
        current_time = time.monotonic()
        last_accepted_event_time = self._last_event_times.get(event_abs_path)

        if last_accepted_event_time and (current_time - last_accepted_event_time) < HANDLER_DEBOUNCE_SECONDS:
            print(f"[ChangeHandler] Debouncing '{mapped_event_type}' event for {event_abs_path} (too close to last accepted event)")
            return # Debounce this event

        # If we've reached here, the event is of a relevant type AND it's not debounced.
        # Record its time as the last accepted event for this path.
        self._last_event_times[event_abs_path] = current_time

        # Clean up old entries from _last_event_times to prevent memory leak
        if len(self._last_event_times) > 1000: # Arbitrary threshold
            cutoff_time = current_time - (HANDLER_DEBOUNCE_SECONDS * 20) # Clean entries older than 20x debounce window
            self._last_event_times = {
                p: t for p, t in self._last_event_times.items() if t > cutoff_time
            }

        # Proceed to queue the event
        try:
            rel_path_posix = event_abs_path.relative_to(self.vault_root).as_posix()
            # This is the key log entry we expect to see for actual processed events
            print(f"[ChangeHandler] Event: {mapped_event_type} on {rel_path_posix} (abs: {event_abs_path})")
            
            fs_event_model = FsEvent(
                kind=mapped_event_type, # type: ignore
                path=rel_path_posix,
            )
            self.queue.put_nowait(fs_event_model)
        except ValueError as e_val: # If path is not within vault_root
            print(f"[ChangeHandler] Ignoring event outside vault root ({type(event_abs_path).__name__} vs {type(self.vault_root).__name__}): {mapped_event_type} on {event_abs_path}")
        except Exception as e: # Includes Pydantic validation errors for FsEvent
            print(f"[ChangeHandler] Error creating FsEvent or queueing for {event_abs_path}: {e}")


class EventBatcher:
    """Batches related file system events."""
    def __init__(self, queue: asyncio.Queue, vault_ctx: VaultCtx):
        self.queue = queue
        self.vault_ctx = vault_ctx
        self.batched_events: Dict[str, Tuple[FsEvent, float]] = {} # path -> (FsEvent, timestamp)
        self.event_order: Deque[str] = deque() # Stores paths in order of arrival for fair processing
        print("[EventBatcher] Initialized.")

    async def run(self):
        print("[EventBatcher] Starting event batching loop...")
        try:
            while True:
                new_event_arrived = False
                try:
                    # Wait for a new event with a timeout
                    event = await asyncio.wait_for(self.queue.get(), timeout=MAX_QUEUE_WAIT_SECONDS)
                    print(f"[EventBatcher] Received from queue: {event.path} ({event.kind})")
                    current_time = time.monotonic()
                    # If it's a delete, it might supersede previous modifications
                    if event.kind == "deleted":
                        self.batched_events[event.path] = (event, current_time)
                    else: # created or modified
                        # If a "create" comes after a "delete" for the same path (quick undelete), let create win.
                        # If "modify" comes, it's the latest state.
                        self.batched_events[event.path] = (event, current_time)

                    if event.path not in self.event_order:
                        self.event_order.append(event.path)
                    self.queue.task_done()
                    new_event_arrived = True
                except asyncio.TimeoutError:
                    # print("[EventBatcher] No new event in this interval.")
                    pass # No new event, proceed to check old events

                now = time.monotonic()
                paths_to_process_this_cycle = []
                
                # Iterate using a copy of deque for safe modification if needed, or use indices
                # Process from oldest:
                processed_count = 0
                for _ in range(len(self.event_order)): # Iterate once over current items
                    path_key = self.event_order.popleft() # Get from left (oldest)
                    
                    if path_key not in self.batched_events: # Could have been processed if logic allows multiple flushes
                        continue

                    fs_event, timestamp = self.batched_events[path_key]
                    age = now - timestamp

                    # Condition to flush:
                    # 1. If a new event arrived this cycle (meaning queue is active) AND this event is old enough (EVENT_BATCH_WINDOW_SECONDS)
                    # OR
                    # 2. If NO new event arrived (queue might be idle) AND this event is very old (PURGE_OLD_EVENTS_SECONDS)
                    should_flush = False
                    if new_event_arrived and age > EVENT_BATCH_WINDOW_SECONDS:
                        should_flush = True
                    elif not new_event_arrived and age > PURGE_OLD_EVENTS_SECONDS: # Flush old items if queue is quiet
                        should_flush = True
                    
                    if should_flush:
                        paths_to_process_this_cycle.append(path_key)
                        processed_count +=1
                    else:
                        self.event_order.append(path_key) # Put back on the right (newest end) if not flushed

                for path_key_to_flush in paths_to_process_this_cycle:
                    if path_key_to_flush in self.batched_events: # Ensure it wasn't processed by another logic path
                        event_bundle, _ = self.batched_events.pop(path_key_to_flush)
                        # Path is already removed from event_order or not added back
                        print(f"[EventBatcher] Flushing event for '{event_bundle.path}' ({event_bundle.kind})")
                        asyncio.create_task(worker_bee(event_bundle, self.vault_ctx))
                
                if not new_event_arrived and not self.batched_events and not self.event_order:
                    await asyncio.sleep(0.1) # brief sleep if totally idle

        except asyncio.CancelledError:
            print("[EventBatcher] Event batching loop cancelled.")
        except Exception as e:
            print(f"[EventBatcher] Critical error in event batching loop: {e}")
            # Consider re-raising or specific handling if it's fatal
        finally:
            print("[EventBatcher] Event batching loop stopped.")


async def worker_bee(event_bundle: FsEvent, vault_ctx: VaultCtx):
    """Processes a single file event: route to agent/handler, run logic, handle patch."""
    print(f"[WorkerBee] Processing event for: {event_bundle.path} ({event_bundle.kind})")

    router = get_router() 
    # route_event is now async and might execute the handler directly or return an agent
    handler_or_status = await router.route_event(event_bundle, vault_ctx)

    if handler_or_status is None:
        print(f"[WorkerBee] No agent or handler found by router for event: {event_bundle.path}. Skipping.")
        return
    
    # Check if the router already handled it (e.g., by calling run_backlinker_for_event)
    if isinstance(handler_or_status, str):
        print(f"[WorkerBee] Router handled event for {event_bundle.path}. Status: {handler_or_status}")
        # If it's a status string, the action was taken by the router (e.g., backlinker orchestrator ran).
        # No further agent processing is needed here for this specific path.
        return
    
    # If we received an Agent instance, proceed with generic agent execution
    if not isinstance(handler_or_status, Agent):
        print(f"[WorkerBee] Router returned an unexpected handler type: {type(handler_or_status)} for {event_bundle.path}. Skipping.")
        return

    agent_to_run: Agent = handler_or_status # Now we know it's an Agent instance

    # File path for operations (relative to vault root from event_bundle.path)
    target_file_path = vault_ctx.root / event_bundle.path
    initial_content_hash = None

    if event_bundle.kind != 'deleted':
        if not target_file_path.is_file():
            print(f"[WorkerBee] File {target_file_path} not found or not a regular file before agent run. Skipping.")
            return
        try:
            content_bytes = target_file_path.read_bytes()
            initial_content_hash = hashlib.md5(content_bytes).hexdigest()
            print(f"[WorkerBee] Pre-run hash for {target_file_path}: {initial_content_hash}")
        except FileNotFoundError:
             print(f"[WorkerBee] File {target_file_path} disappeared just before reading for hash. Skipping.")
             return
        except Exception as e:
            print(f"[WorkerBee] Error reading file {target_file_path} for pre-run check: {e}. Skipping.")
            return

    print(f"[WorkerBee] Running agent '{agent_to_run.__class__.__name__}' for {event_bundle.path}")
    patch_or_response = None
    try:
        # This user_message is now only for GENERIC agents returned by the router.
        # The backlinker agent is no longer called this way.
        user_message = (
            f"The file '{event_bundle.path}' (kind: {event_bundle.kind}) has changed. "
            f"Please analyze its content and take appropriate action based on your configured role. "
            f"If your role involves modifying this file, return a Patch object. "
            f"Otherwise, you might return another type of response or data structure."
        )

        patch_or_response = await agent_to_run.run(user_message, deps=vault_ctx)
    except Exception as e:
        print(f"[WorkerBee] Error running agent {agent_to_run.__class__.__name__} for {event_bundle.path}: {e}")
        if vault_ctx.kv:
            try:
                vault_ctx.kv.table("agent_errors").insert({
                    "timestamp": time.time(), "agent": agent_to_run.__class__.__name__,
                    "event_path": event_bundle.path, "error": str(e)
                })
            except Exception as kv_e:
                print(f"[WorkerBee] CRITICAL: Failed to log agent error to KV store: {kv_e}")
        return

    # The rest of the patch handling logic from the original worker_bee remains largely the same.
    # It assumes that if an agent is run, it might return a Patch object.
    # This part needs to be reviewed if agents can return other things, but for now, Patch is expected.

    if not isinstance(patch_or_response, Patch):
        print(f"[WorkerBee] Agent {agent_to_run.__class__.__name__} did not return a Patch object. Response: {patch_or_response}")
        if vault_ctx.kv: 
            try:
                vault_ctx.kv.table("agent_direct_responses").insert({
                    "timestamp": time.time(), "agent": agent_to_run.__class__.__name__,
                    "event_path": event_bundle.path, "response": str(patch_or_response)
                })
            except Exception as kv_e:
                print(f"[WorkerBee] CRITICAL: Failed to log agent direct response to KV store: {kv_e}")
        return

    patch: Patch = patch_or_response
    # The existing Patch model has target_path and action, but not event_path, before, after, comment.
    # The Patch model from `models.py` needs to be aligned with what `run_backlinker_for_event`'s agent produces
    # and what `tools_common.write_patch` expects.
    # The current Patch model in `models.py` is: class Patch(BaseModel): action: str; target_path: str; content: str; event_path: str;
    # The Patch used by backlinker agent has: before, after, comment, agent.
    # This is a MISMATCH that needs to be resolved. For now, proceeding with assumption that patch object has a target_path and action.
    # The `write_patch` in `tools_common` expects a Patch with `before`, `after`, `comment`, `agent`.
    # The generic part of worker_bee below might be incompatible if `patch.target_path` or `patch.action` are not on the Patch from backlinker.
    # Let's assume for now the backlinker *doesn't* go through this generic patch processing section if it was handled by the router.
    # This section below IS for agents that are NOT the backlinker and are run directly by worker_bee.

    print(f"[WorkerBee] Agent {agent_to_run.__class__.__name__} proposed patch for target '{patch.target_path if hasattr(patch, 'target_path') else event_bundle.path}' with action '{patch.action if hasattr(patch, 'action') else 'unknown'}'")

    patch_target_abs_path = vault_ctx.root / (patch.target_path if hasattr(patch, 'target_path') else event_bundle.path)

    # Check if `patch` has `action` and `content` attributes, as the generic part expects.
    if not hasattr(patch, 'action') or not hasattr(patch, 'content'):
        print(f"[WorkerBee] Patch object from agent {agent_to_run.__class__.__name__} is missing 'action' or 'content'. Cannot apply. Patch: {patch}")
        return

    # For actions that modify an existing file (like APPEND or custom ones),
    # or actions where the target isn't clear until after the agent runs (potentially),
    # we first check if the target file (if it's the event source) has changed content or disappeared.
    print(f"DEBUG_WB: Before L310 conditional block: target='{patch_target_abs_path}', exists='{patch_target_abs_path.exists()}', is_file='{patch_target_abs_path.is_file()}'") # DEBUG
    if patch.action not in ["CREATE", "DELETE"]: 
        if not patch_target_abs_path.is_file():
            print(f"[WorkerBee] Target file '{patch_target_abs_path}' for patch disappeared or is not a file. Skipping patch.")
            if vault_ctx.kv: 
                vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "target_disappeared", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
            return
        if hasattr(patch, 'target_path') and patch.target_path == event_bundle.path and event_bundle.kind != 'deleted':
            print(f"DEBUG_WB: Inside L316 condition, before L317 try: target='{patch_target_abs_path}', exists='{patch_target_abs_path.exists()}', is_file='{patch_target_abs_path.is_file()}'") # DEBUG
            try:
                print(f"DEBUG_WB: Before L318 read_bytes: patch_target_abs_path='{patch_target_abs_path}', exists='{patch_target_abs_path.exists()}', is_file='{patch_target_abs_path.is_file()}'") # DEBUG
                current_content_bytes_after_agent = patch_target_abs_path.read_bytes()
                current_hash_after_agent = hashlib.md5(current_content_bytes_after_agent).hexdigest()
                if current_hash_after_agent != initial_content_hash:
                    print(f"[WorkerBee] File '{patch_target_abs_path}' content changed during agent run. Skipping patch.")
                    if vault_ctx.kv: 
                        vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "target_modified_during_run", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
                    return
            except FileNotFoundError:
                print(f"[WorkerBee] Target file '{patch_target_abs_path}' not found during pre-patch hash check. Skipping.")
                if vault_ctx.kv: 
                    vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "target_disappeared_pre_hash", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
                return
            except Exception as e:
                print(f"[WorkerBee] Error re-reading file '{patch_target_abs_path}' for pre-patch check: {e}. Skipping patch.")
                if vault_ctx.kv: 
                    vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "target_read_error_pre_patch", "error": str(e), "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
                return

    # Check if the target file was recently modified by an agent to prevent loops
    # This applies if the target path exists (for any action) or if it's a CREATE action (path might not exist yet)
    effective_initial_hash = initial_content_hash if patch_target_abs_path.exists() else None
    if (patch_target_abs_path.exists() or patch.action == "CREATE") and \
       vault_ctx.was_recently_modified_by_agent(patch_target_abs_path, effective_initial_hash):
        print(f"[WorkerBee] File {patch_target_abs_path} was recently modified by an agent (or is a CREATE for a recently touched path). Skipping patch to prevent loops.")
        if vault_ctx.kv: 
            vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "recently_modified_skip", "patch_target": str(patch_target_abs_path), "initial_hash_provided": effective_initial_hash is not None})
        return

    # Action-specific pre-conditions before attempting the patch action
    if patch.action == "CREATE" and patch_target_abs_path.exists():
        print(f"[WorkerBee] Patch action CREATE, but file '{patch_target_abs_path}' already exists. Skipping to avoid overwrite.")
        if vault_ctx.kv: 
            vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "create_target_exists", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
        return
    elif patch.action == "DELETE" and not patch_target_abs_path.is_file():
         print(f"[WorkerBee] Patch action DELETE, but file '{patch_target_abs_path}' does not exist or not a file. Skipping.")
         if vault_ctx.kv:
             vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "delete_target_not_exist_or_not_file", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
         return
    
    try:
        if patch.action == "APPEND":
            print(f"[WorkerBee] Appending to {patch_target_abs_path}")
            with patch_target_abs_path.open("a", encoding="utf-8") as f:
                f.write(patch.content)
        elif patch.action == "CREATE":
            print(f"[WorkerBee] Creating file {patch_target_abs_path}")
            patch_target_abs_path.parent.mkdir(parents=True, exist_ok=True)
            patch_target_abs_path.write_text(patch.content, encoding="utf-8")
        elif patch.action == "DELETE":
            print(f"[WorkerBee] Deleting file {patch_target_abs_path}")
            patch_target_abs_path.unlink()
        else:
            print(f"[WorkerBee] Unknown or unhandled patch action for generic agent: '{patch.action}'. Not applying.")
            if vault_ctx.kv: 
                vault_ctx.kv.table("patch_aborted").insert({"timestamp": time.time(), "reason": "unknown_patch_action_generic", "patch": patch.model_dump() if hasattr(patch, 'model_dump') else str(patch)})
            return

        print(f"[WorkerBee] Generic patch applied successfully for '{patch_target_abs_path}' by {agent_to_run.name}")
        if vault_ctx.kv:
            event_path_for_log = patch.event_path if hasattr(patch, 'event_path') else event_bundle.path
            vault_ctx.kv.table("applied_patches").insert({
                "timestamp": time.time(), "agent": agent_to_run.name,
                "patch_event_path": event_path_for_log,
                "patch_target_path": (patch.target_path if hasattr(patch, 'target_path') else event_bundle.path),
                "patch_action": patch.action,
            })
    except Exception as e:
        print(f"[WorkerBee] Error applying generic patch to '{patch_target_abs_path}': {e}")
        if vault_ctx.kv:
            target_path_for_log = patch.target_path if hasattr(patch, 'target_path') else event_bundle.path
            vault_ctx.kv.table("failed_patches").insert({
                "timestamp": time.time(), "agent": agent_to_run.name,
                "patch_target": target_path_for_log, "patch_action": patch.action, "error": str(e)
            })


async def run_observer(vault_path_str: str, vault_ctx: VaultCtx):
    event_queue = asyncio.Queue()
    # Ensure vault_ctx.root is a Path object for ChangeHandler
    event_handler = ChangeHandler(event_queue, Path(vault_ctx.root), vault_ctx)
    
    batcher = EventBatcher(event_queue, vault_ctx)
    batcher_task = asyncio.create_task(batcher.run(), name="EventBatcherTask")

    observer = Observer()
    observer.schedule(event_handler, vault_path_str, recursive=True)
    
    # Observer runs in a separate thread, so starting it is non-blocking for asyncio
    observer.start()
    print(f"[Observer] Started watching directory: {vault_path_str}")

    try:
        # Keep this main coroutine for run_observer alive.
        # The batcher_task will run in the background, and it spawns worker_bee tasks.
        while True:
            await asyncio.sleep(3600) # Sleep for a long time, or use an asyncio.Event for shutdown
    except asyncio.CancelledError:
        print("[Observer] run_observer task itself cancelled.")
        # This would typically be initiated from main_async_runner's finally block or KeyboardInterrupt
    # KeyboardInterrupt should ideally be caught in main_async_runner to initiate graceful shutdown.
    # If it's caught here, it means main_async_runner didn't catch it first.
    except KeyboardInterrupt:
        print("[Observer] KeyboardInterrupt caught directly in run_observer. Initiating shutdown sequence.")
        # This is a fallback; main_async_runner should ideally handle this.
    finally:
        print("[Observer] run_observer stopping components...")
        if observer.is_alive():
            observer.stop()
            print("[Observer] Observer stop() called.")
            # observer.join() can block, which is problematic in an async finally.
            # It's better if main_async_runner manages the join after all async tasks are cancelled.
            # For now, we signal stop and rely on thread termination.
        
        if batcher_task and not batcher_task.done():
            batcher_task.cancel()
            try:
                await batcher_task # Give it a chance to clean up
            except asyncio.CancelledError:
                print("[Observer] Batcher task successfully cancelled during shutdown.")
            except Exception as e: # Catch any other error during batcher cancellation
                print(f"[Observer] Error awaiting cancelled batcher task: {e}")
        
        # Ensure observer thread is really finished after tasks that might use it are done.
        if observer.is_alive():
            observer.join(timeout=5.0) # Join with a timeout
            if observer.is_alive():
                print("[Observer] Warning: Observer thread did not join cleanly after 5s timeout.")
            else:
                print("[Observer] Observer thread joined successfully.")
        print("[Observer] run_observer cleanup attempted.") 