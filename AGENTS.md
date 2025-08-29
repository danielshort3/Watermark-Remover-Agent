# Repository Guidelines

## Project Structure & Modules
- `src/`: Python sources.
  - `watermark_remover/agent/`: LangChain agent, tools, and LangGraph graphs (`single_song_graph.py`, `order_of_worship_graph.py`).
  - `models/`: UNet/VDSR and helpers.
  - `utils/`: Selenium/XPath and transposition utilities.
  - `config/settings.py`: Defaults (e.g., `DEFAULT_OLLAMA_URL`, `DEFAULT_OLLAMA_MODEL`).
- `tests/`: Unit tests (`test_*.py`).
- `scripts/`: Developer scripts (e.g., `run_agent.py`).
- Data-only: `data/`, `models/` (weights), `input/`, `output/` (results). Avoid committing large binaries.
- `langgraph.json`: Graph entry points for `langgraph dev`.

## Build, Test, and Development
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Tests: `pytest -q` (or `python -m unittest`).
- Run once: `python scripts/run_agent.py "Download 'Fur Elise' for Horn in F and make a PDF"`.
- Agent REPL: `python -m watermark_remover.agent.ollama_agent`.
- LangGraph server (from repo root): `langgraph dev --host 0.0.0.0 --port 2024` (uses `langgraph.json`).
- Docker (GPU optional): `docker build -t watermark-remover-agent .` then run with `OLLAMA_URL`/`OLLAMA_MODEL` envs.

## Coding Style & Naming
- Python, PEP 8, 4-space indent, type hints required for new/changed code.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Docstrings: concise triple-quoted summaries; avoid import-time side effects.
- Use `logging`; avoid `print`. Keep functions small and testable.

## Testing Guidelines
- Framework: `pytest` (works with `unittest`). Tests live in `tests/` and are named `test_*.py`.
- Keep tests hermetic: use temp dirs, no network, no external state; heavy Torch paths should be optional/skipped when unavailable.
- Add/adjust tests for any new tool, model helper, or graph logic.

## Commit & Pull Request Guidelines
- Commits: imperative present (“Add tool …”), small and focused; prefixes like `feat:`, `fix:` are welcome.
- PRs must include: summary (what/why), key changes, how to test (commands/expected output), and linked issues.
- Update docs (`README.md`, this file) when adding tools/graphs or changing interfaces. Don’t commit large assets or secrets.

## Configuration & Agent Tips
- Environment: `OLLAMA_URL` (default local), `OLLAMA_MODEL` (e.g., `qwen3:30b`), `LOG_LEVEL`, optional `WMRA_LOG_DIR`.
- Weights live under `models/`; sample sheets under `data/samples/`.
- Add a tool: implement in `src/watermark_remover/agent/tools.py` (clear docstring), ensure it’s imported/used by the agent, and add tests.
- Add a graph: create `src/watermark_remover/agent/<name>_graph.py` exposing `graph`, register in `langgraph.json`, and verify via `langgraph dev`.

