# Obsidian Vault Agent with Pydantic AI

This project implements a basic framework for creating agents that interact with your Obsidian vault, leveraging the Pydantic AI library for structured interactions with Large Language Models (LLMs).

## Features (Conceptual - based on initial setup)

*   **File System Monitoring**: Uses `watchdog` to monitor changes in your Obsidian vault.
*   **Event Batching**: Groups rapid file modifications to avoid redundant processing.
*   **Agent Routing**: A simple router (`router.py`) to direct file events to specific agents based on path patterns.
*   **Pydantic AI Integration**: Agents (`agents/`) are built using `pydantic-ai`, allowing for typed inputs/outputs, tool usage, and interaction with LLMs.
*   **Dependency Injection**: A `VaultCtx` (`deps.py`) provides agents with access to vault configuration, database connections (DuckDB for embeddings/metadata), and a key-value store (TinyDB for logging).
*   **Example Agent**: A `BacklinkerAgent` (`agents/backlinker.py`) is provided as a starting point, designed to suggest backlinks for notes (though its core LLM logic and embedding search are placeholders).

## Project Structure

```
.
├── agents/                 # Agent implementations
│   ├── __init__.py
│   └── backlinker.py       # Example backlinking agent
├── .venv/                  # Virtual environment (if you create one here)
├── config.yaml             # Configuration file for vault path, DB, etc.
├── deps.py                 # Defines VaultCtx for dependency injection
├── main.py                 # Main entry point to run the agent system
├── models.py               # Pydantic models for events (FsEvent, Patch)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── router.py               # Routes events to appropriate agents
├── runner.py               # Core event loop, file watcher, and agent execution logic
└── tools_common.py         # Common tools usable by multiple agents
```

## Setup

1.  **Clone the Repository** (or create the files as per the transcript if this is generated code).

2.  **Create and Activate a Virtual Environment**:
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # On Windows
    # .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies using `uv`**:
    If you don't have `uv`, install it first (see [uv documentation](https://github.com/astral-sh/uv)).
    ```bash
    uv pip install -r requirements.txt
    ```
    You might also need to install specific extras for `pydantic-ai` depending on the LLM provider you intend to use (e.g., `pydantic-ai[ollama]` or `pydantic-ai[openai]`). Add this to `requirements.txt` or install manually.

4.  **Configure Your LLM Provider**:
    *   Pydantic AI needs to know how to connect to your LLM. This is often done via environment variables (e.g., `OPENAI_API_KEY` for OpenAI, or `PYDANTIC_AI_OLLAMA_HOST` for a local Ollama instance).
    *   The example `agents/backlinker.py` uses a model string like `"ollama/mistral:7b-instruct"`. Ensure your chosen LLM (e.g., Ollama) is running and has the specified model downloaded.
    *   The `main.py` script attempts to set `PYDANTIC_AI_OLLAMA_HOST` if not already set. Adjust this as needed for your setup.

5.  **Configure the Agent**: 
    *   Copy or rename `config.yaml.example` to `config.yaml` if an example is provided, or create `config.yaml`.
    *   Edit `config.yaml` to set `vault_path` to the correct path of your Obsidian vault.
    *   Adjust database (`obsidian_agent.db`) and KV store (`agent_kv_log.json`) names/paths if desired. These will be created inside your vault path by default.

## Running the Agent

Once set up, you can run the agent system from the root directory of the project:

```bash
python main.py
```

The script will:
1.  Load configuration from `config.yaml`.
2.  Initialize the `VaultCtx` (database connections, etc.).
3.  Start monitoring the specified Obsidian vault path for file changes.
4.  When a file is modified or created, the event will be processed, routed to an agent (currently, `.md` files go to the `BacklinkerAgent`), and the agent will attempt to run.

## Development

*   **Adding New Agents**: 
    1.  Create a new Python file in the `agents/` directory (e.g., `agents/digest_agent.py`).
    2.  Define your agent using `pydantic_ai.Agent`, specifying `deps_type=VaultCtx` and an appropriate `output_type` (likely `Patch` from `models.py` or a custom model).
    3.  Implement tools for your agent using the `@agent.tool` decorator.
    4.  Add rules to `router.py` to direct events to your new agent.
*   **Tools**: Common tools can be placed in `tools_common.py` and imported by agents.
*   **Debugging**: Look at the console output for logs from the runner, handler, and agents. Pydantic AI's `capture_run_messages` can be helpful for debugging LLM interactions (see commented-out sections in `runner.py`).

## Important Notes & Caveats

*   **LLM Costs/Quotas**: If using cloud-based LLMs (like OpenAI), be mindful of API costs and rate limits.
*   **Local LLMs**: For local LLMs (Ollama, llama.cpp), ensure they are running and accessible. Performance will depend on your hardware.
*   **Error Handling**: The current error handling is basic. Robust production systems would need more comprehensive error management, retries, and dead-letter queues for events.
*   **Idempotency**: The `BacklinkerAgent` and patching mechanism are placeholders. True idempotent operations (applying the same change multiple times having no new effect) are crucial and require careful design, especially with LLM-generated content.
*   **Security**: Be cautious, especially if agents can write files or execute commands. The current setup is for local execution and assumes you trust the code and the LLM outputs to a reasonable extent within your own vault.
*   **Atomic Writes**: The `runner.py` attempts atomic writes for patches (write to `.tmp` then replace) to minimize issues with partial writes being picked up by the watcher.

## Next Steps / Potential Enhancements

*   Implement the actual embedding generation and similarity search for `list_similar_notes` in `agents/backlinker.py`.
*   Develop more sophisticated agents from the ideas list.
*   Add a UI or better CLI for managing agents and viewing logs.
*   Integrate with Obsidian directly via a plugin for a richer experience.
*   Implement robust testing.
