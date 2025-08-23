# Watermark Remover Agent (Modified)

This repository contains a modified version of the Watermark‑Remover agent.
Key changes include:

* **No stub dependencies** – The `langchain`, `langchain_core` and `langgraph` stub
  packages have been removed from this repo.  You must install the real
  `langchain`, `langchain_ollama`, and `langgraph` packages in your
  environment (the provided Dockerfile does this automatically).
* **Centralised logging** – All logs and screenshots are written into
  `logs/<timestamp>` in the project root.  You can mount this directory
  with `-v $(pwd)/output:/app/logs` to collect logs on the host.  The
  log directory is also exposed to the scraper and agent via the
  `WMRA_LOG_DIR` environment variable.
* **Improved scraper** – The `scrape_music` tool no longer falls back
  to `data/samples`.  It iterates through search results on
  PraiseCharts, selects the requested key and instrument (or a
  fallback) and downloads preview images.  If a candidate lacks
  orchestration, the scraper moves on to the next candidate rather
  than looping.  Fallback selections and reasons are recorded in
  the logs.
* **Reasoning capture** – The `run_instruction` helper in
  `graph_ollama.py` captures the LLM’s reasoning along with the final
  answer and writes it to `thoughts_and_steps.log` under the log
  directory.  This allows you to inspect why a different key or
  instrument was chosen.

See the individual module docstrings for more details on the tools
and helper functions.