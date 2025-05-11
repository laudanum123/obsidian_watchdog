import re
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Type, Dict, Any, Coroutine
from pydantic_ai import Agent
from pydantic import BaseModel
from models import VaultConfig, Patch, FsEvent
from deps import VaultCtx

# Import agents
from agents import backlinker # Assuming backlinker.py contains 'backlink_agent'
# from .agents import digest, flashcard, duplicate_finder # Example for future agents

# Agent specific imports - these would ideally be discovered or registered
# For now, we explicitly import the agents we know about.
from agents.backlinker import backlink_agent, run_backlinker_for_event # Assuming backlink_agent is the one to run


# Define the structure for an agent instance more explicitly if needed
# For now, Pydantic AI Agent itself is generic enough.

# RULES: List of tuples (regex_pattern, agent_instance_or_factory)
# The agent factory could be a function that returns an initialized agent if complex setup is needed per call.
# RULES: List[Tuple[re.Pattern, Agent]] = [
#     # Example: Route all .md files to the backlink_agent
#     (re.compile(r"\.md$", re.IGNORECASE), backlinker.backlink_agent),
#     # Add more rules for other agents as they are developed:
#     # (re.compile(r"^Daily/", re.IGNORECASE), digest.daily_agent), 
#     # (re.compile(r"^Knowledge/", re.IGNORECASE), flashcard.card_agent),
#     # (re.compile(r".*", re.IGNORECASE), duplicate_finder.dup_agent), # Default/fallback agent
# ]

# def choose_agent(relative_path: str) -> Optional[Agent]: # This function seems unused, can be removed or kept if planned for future
#     """
#     Selects an agent based on the relative path of the modified file.
#     """
#     # ... (implementation can be removed if RULES is removed and this is not used)
#     return None 

class Router:
    def __init__(self):
        self.routes: List[Tuple[re.Pattern, Any]] = [ # Changed Agent to Any to allow functions
            (re.compile(r"\.md$", re.IGNORECASE), run_backlinker_for_event), # Route to the orchestrator function
            # Example for future agents that might still return an Agent instance:
            # (re.compile(r"other_pattern", re.IGNORECASE), some_other_agent_instance),
        ]
        print("[Router] Initialized with defined routes.")

    async def route_event(self, event: FsEvent, vault_ctx: VaultCtx) -> Optional[Any]: # Made async, return type Any
        """
        Routes an event. For backlinker, it directly calls the orchestrator.
        For others, it might return an agent instance.
        """
        for pattern, handler in self.routes:
            if pattern.search(str(event.path)): 
                if handler == run_backlinker_for_event:
                    print(f"[Router] Event for '{event.path}' matches backlinker. Calling orchestrator...")
                    try:
                        await run_backlinker_for_event(
                            event_path=str(event.path), # run_backlinker_for_event expects str
                            event_type=event.kind, # Changed from event.event_type to event.kind
                            vault_ctx=vault_ctx
                        )
                        return "ran_backlinker_orchestrator" # Indicate action taken
                    except Exception as e:
                        print(f"[Router] Error calling run_backlinker_for_event for '{event.path}': {e}")
                        import traceback
                        traceback.print_exc()
                        return "backlinker_orchestrator_error"
                elif isinstance(handler, Agent): # For other potential agents
                    print(f"[Router] Routing event for '{event.path}' to agent: {handler.model}")
                    return handler # Return agent instance for worker bee to handle
                else:
                    print(f"[Router] Error: Matched route for '{event.path}' but found unhandled handler type: {type(handler)}")
                    return None
        
        print(f"[Router] No agent/handler found for event: {event.path}")
        return None

# Global router instance (optional, could be managed by a DI container)
_router_instance = None

def get_router() -> Router: # This function might need to be async if Router init becomes async
    """Returns a global instance of the Router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
    return _router_instance 