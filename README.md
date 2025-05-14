# Obsidian Vault Agent with Pydantic AI

This project implements a basic framework for creating agents that interact with your Obsidian vault, leveraging the Pydantic AI library for structured interactions with Large Language Models (LLMs).

## Features (Conceptual - based on initial setup)

*   **File System Monitoring**: Uses `watchdog` to monitor changes in your Obsidian vault.
*   **Event Batching**: Groups rapid file modifications to avoid redundant processing.
*   **Agent Routing**: A simple router (`src/obsidian_watchdog/router.py`) to direct file events to specific agents.
*   **Pydantic AI Integration**: Agents (e.g., in `src/obsidian_watchdog/agents/`) are built using `pydantic-ai` for structured LLM interactions.
*   **Dependency Injection**: A `VaultCtx` (`src/obsidian_watchdog/deps.py`) provides agents with access to vault configuration and database connections.
*   **Database Management (`src/obsidian_watchdog/notes_db_manager.py`)**:
    *   Manages a DuckDB database for note metadata and embeddings.
    *   Supports schema versioning (e.g., includes `last_backlinked_at` to track agent processing).
    *   Chunks notes into smaller segments for more granular embeddings.
    *   Generates embeddings using a configured OpenAI-compatible API.
    *   Builds an HNSW index for efficient vector similarity searches.
    *   Prunes data for notes that are deleted from the vault.
*   **Example Agent - `BacklinkerAgent` (`src/obsidian_watchdog/agents/backlinker.py`)**:
    *   Identifies potentially related notes using embedding similarity search (via HNSW index).
    *   Uses an LLM (via Pydantic AI and a `LinkDecision` model) to decide if a wikilink to a similar note should be added.
    *   Checks for pre-existing links to avoid duplicates.
    *   If approved by the LLM, adds the new wikilink(s) under a dedicated "## Backlinks" section at the end of the original note.
    *   Can run in batch mode to process all notes needing updates (e.g., on startup or if modified since last backlinking).
    *   Updates a `last_backlinked_at` timestamp in the database for processed notes.

## Project Structure

```
.
├── .venv/                  # Virtual environment
├── src/                    # Source code
│   └── obsidian_watchdog/
│       ├── __init__.py
│       ├── agents/             # Agent implementations
│       │   ├── __init__.py
│       │   └── backlinker.py   # Example backlinking agent
│       ├── deps.py             # Defines VaultCtx for dependency injection
│       ├── main.py             # Main entry point to run the agent system
│       ├── models.py           # Pydantic models for events (FsEvent, Patch)
│       ├── notes_db_manager.py # Manages the DuckDB database for notes & embeddings
│       ├── router.py           # Routes events to appropriate agents
│       ├── runner.py           # Core event loop, file watcher, and agent execution logic
│       └── tools_common.py     # Common tools usable by multiple agents
├── config.yaml             # Configuration file for vault path, DB, etc.
├── pyproject.toml          # Python project configuration and dependencies
├── README.md               # This file
└── ...                     # Other project files (.gitignore, etc.)
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
    uv pip install -e .
    ```
    This command installs the project in editable mode along with its dependencies defined in `pyproject.toml`.
    You might also need to install specific extras for `pydantic-ai` depending on the LLM provider you intend to use (e.g., `pydantic-ai[ollama]` or `pydantic-ai[openai]`). You can define these as extras in `pyproject.toml` (e.g., `obsidian-watchdog[ollama]`) or install them manually.
    The `BacklinkerAgent` example uses an OpenAI-compatible API, which can be pointed to services like LM Studio.

4.  **Configure Your LLM Provider**:
    *   Pydantic AI needs to know how to connect to your LLM. This is often done via environment variables (e.g., `OPENAI_API_KEY` for OpenAI, `PYDANTIC_AI_OLLAMA_HOST` for a local Ollama instance).
    *   The example `src/obsidian_watchdog/agents/backlinker.py` is configured to use a local LM Studio instance by default (e.g., `base_url='http://<your-lmstudio-ip>:1234/v1'`). Ensure your chosen LLM is running and has the specified model downloaded.
    *   The `src/obsidian_watchdog/main.py` script might attempt to set some default environment variables if not present (e.g., `PYDANTIC_AI_OLLAMA_HOST`). Adjust as needed.

5.  **Configure the Agent (`config.yaml`)**:
    *   Copy or rename `config.yaml.example` to `config.yaml` if an example is provided, or create `config.yaml`.
    *   Edit `config.yaml` to set:
        *   `vault_path`: Correct path to your Obsidian vault.
        *   `db_path` (optional): Path to the DuckDB database file (e.g., `your_vault/.obsidian_watchdog/notes.db`).
        *   `kv_log_path` (optional): Path to the TinyDB key-value log file.
        *   `embedding_model`:
            *   `model_name`: Name of the embedding model (e.g., `text-embedding-mxbai-embed-large-v1` or as required by your provider).
            *   `dimensions`: The output dimension of your embedding model (e.g., 1024). This is crucial for DB schema and HNSW index.
        *   `db_population`:
            *   `embedding_batch_size`: How many text chunks to send for embedding in one API call.
            *   `embedding_chunk_size`: Target size (in characters) for splitting notes into chunks before embedding.
            *   `force_reembed_all` (boolean, optional): Set to `true` to re-embed all notes on next startup, ignoring hashes and timestamps. Useful after changing embedding models or chunking strategy.
            *   `ignore_patterns`: List of path prefixes to ignore when scanning the vault.
    *   The database and KV store will typically be created inside your vault path (e.g., in a subdirectory like `.obsidian_watchdog/`) if relative paths are used or defaults are kept.

## Running the Agent

Once set up, you can run the agent system from the root directory of the project:

```bash
python -m obsidian_watchdog.main
```

The script will:
1.  Load configuration from `config.yaml`.
2.  Initialize `VaultCtx` and the DuckDB database.
3.  Run `populate_notes_db` from `src/obsidian_watchdog/notes_db_manager.py`:
    *   Scans the vault for markdown files.
    *   Identifies new or modified notes (or all if `force_reembed_all` is true).
    *   Chunks content, generates embeddings, and stores them in DuckDB.
    *   Builds/updates the HNSW index for similarity search.
    *   Prunes deleted notes from the database.
4.  Start monitoring the specified Obsidian vault path for file changes.
5.  When a file is modified or created:
    *   The event is processed by `runner.py`.
    *   The event is routed by `router.py` to an agent (e.g., `.md` files to `BacklinkerAgent`).
    *   The `BacklinkerAgent` (`run_backlinker_for_event` in `src/obsidian_watchdog/agents/backlinker.py`):
        *   Finds similar notes using embeddings.
        *   Queries an LLM to decide whether to add a link.
        *   If approved, generates a `Patch` to add the link under a "## Backlinks" section in the original note.
        *   The patch is applied, and `last_backlinked_at` is updated.
    *   The system may also run a batch backlinking process (e.g., via `run_backlinker_for_all_notes`) for notes that haven't been processed recently.

## Development

*   **Adding New Agents**: 
    1.  Create a new Python file in the `src/obsidian_watchdog/agents/` directory (e.g., `src/obsidian_watchdog/agents/digest_agent.py`).
    2.  Define your agent using `pydantic_ai.Agent`, specifying `deps_type=VaultCtx` and an appropriate `output_type` (likely `Patch` from `src/obsidian_watchdog/models.py` or a custom model).
    3.  Implement tools for your agent using the `@agent.tool` decorator.
    4.  Add rules to `src/obsidian_watchdog/router.py` to direct events to your new agent.
*   **Tools**: Common tools can be placed in `src/obsidian_watchdog/tools_common.py` and imported by agents.
*   **Debugging**: Look at the console output for logs from the runner, handler, and agents. Pydantic AI's `capture_run_messages` can be helpful for debugging LLM interactions (see commented-out sections in `src/obsidian_watchdog/runner.py`).

## Important Notes & Caveats

*   **LLM Costs/Quotas**: If using cloud-based LLMs (like OpenAI), be mindful of API costs and rate limits.
*   **Local LLMs**: For local LLMs (Ollama, llama.cpp), ensure they are running and accessible. Performance will depend on your hardware.
*   **Error Handling**: The current error handling is basic. Robust production systems would need more comprehensive error management, retries, and dead-letter queues for events.
*   **Idempotency**: The `BacklinkerAgent` aims for better idempotency by checking existing links and managing a "## Backlinks" section. However, LLM interactions can still introduce variability. Careful design of prompts and post-processing is key.
*   **Security**: Be cautious, especially if agents can write files or execute commands. The current setup is for local execution and assumes you trust the code and the LLM outputs to a reasonable extent within your own vault.
*   **Atomic Writes**: The `src/obsidian_watchdog/runner.py` attempts atomic writes for patches (write to `.tmp` then replace) to minimize issues with partial writes being picked up by the watcher.

## Next Steps / Potential Enhancements

*   Implement more sophisticated embedding strategies (e.g., different chunking, multi-vector).
*   Develop more sophisticated agents from the ideas list (e.g., summarization, question-answering over notes).
*   Add a UI or better CLI for managing agents and viewing logs.
*   Integrate with Obsidian directly via a plugin for a richer experience.
*   Implement robust testing.
