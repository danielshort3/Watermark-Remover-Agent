# Watermark Remover Agent (Modified)

This repository revives the **Watermark‑Remover‑Agent** with full PyTorch
support and a locally hosted LLM (via Ollama) to orchestrate a multi‑step
pipeline.  The agent can scrape (stubbed in this example), remove
watermarks, upscale images, and assemble the results into a PDF—all
without manual intervention.  Instead of wiring these steps together via
LangGraph, the project uses a **LangChain** agent backed by an Ollama
model (e.g. `qwen3:30b`) to decide which tool to call and when.

## Prerequisites

* **GPU‑enabled Docker**: The provided `Dockerfile` uses the
  `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` base image.  Your
  system must support `--gpus all` for CUDA acceleration, though the code
  will fall back to CPU if no GPU is available.
* **Ollama server**: Start an Ollama server on the host and pull the
  desired model.  For example:

  ```bash
  ollama serve &
  ollama pull qwen3:30b
  ```

## Building the image

The heavy dependencies (torch, torchvision, pytorch‑msssim, opencv,
reportlab, etc.) are declared in `requirements.txt`.  To build the
container run:

```bash
docker build -t watermark-remover-agent .
```

To run with GPU access and connect to an Ollama server running on the
host machine, use a command similar to:

```bash
docker run --gpus all --rm \
  -p 2024:2024 \
  --add-host=host.docker.internal:host-gateway \
  -e LOG_LEVEL=DEBUG \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen3:30b \
  -e OLLAMA_KEEP_ALIVE=30m \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  watermark-remover-agent
```

When the container starts you can either run an interactive REPL via
`python -m watermark_remover.agent.ollama_agent` or execute a single
instruction with the convenience function described below.

Example user input (JSON payload):

```json
{"user_input": "Download the songs in 'October 12, 2025.pdf' for the French Horn."}
```

## Using the agent

You can create a reusable agent object by calling
`get_ollama_agent`.  This requires that `langchain`, `langchain‑ollama`
and all heavy dependencies are installed (they will be when using the
provided Dockerfile).  For example:

```python
from watermark_remover.agent.ollama_agent import get_ollama_agent

# Create an agent bound to the Qwen3 model on the Ollama server
agent = get_ollama_agent(model_name="qwen3:30b")

# Ask the agent to download some sheet music, remove the watermark,
# upscale it and assemble a PDF.  The agent will call the appropriate
# tools defined in watermark_remover.agent.tools.
response = agent.invoke({"input": "Download Fur Elise sheet music, remove the watermark, upscale it, and assemble into a PDF."})
print(response)
```

If you don't need to reuse the agent, use the convenience wrapper
`run_instruction` to perform a single task:

```python
from watermark_remover.agent.graph_ollama import run_instruction

result = run_instruction("Download Fur Elise sheet music, remove the watermark, upscale it and assemble into a PDF.")
print(result)
```

Both methods return a string with the result or a diagnostic error if
the Ollama server cannot be reached or the model is missing.

## Project Structure (src‑layout)

The repo follows a clean `src/` layout with clear boundaries between application code, utilities, models, tests, and scripts.

```
.
├─ src/
│  ├─ watermark_remover/
│  │  ├─ __init__.py
│  │  └─ agent/
│  │     ├─ __init__.py
│  │     ├─ graph_ollama.py
│  │     ├─ ollama_agent.py
│  │     ├─ single_song_graph.py
│  │     └─ order_of_worship_graph.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ selenium_utils.py
│  │  └─ transposition_utils.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ model_functions.py  # UNet, VDSR, PIL/tensor helpers
│  └─ config/
│     ├─ __init__.py
│     └─ settings.py       # default OLLAMA URL/MODEL, log level
├─ tests/
│  ├─ test_agent_tools.py
│  └─ test_graph_ollama.py
├─ scripts/
│  └─ run_agent.py
├─ data/
│  └─ README.md
├─ models/                 # weights/checkpoints live here (data only)
├─ input/                  # sample inputs (if any)
├─ output/                 # logs/results
├─ langgraph.json          # depends on ./src for module imports
├─ requirements.txt
└─ Dockerfile
```

Notes:
- Python modules live under `src/`. Tests add `src/` to `sys.path` for direct invocation.
- ML/DL code is in `src/models/model_functions.py` and imported as `models.*`.
- Utilities live in `src/utils/` and are imported as `utils.*`.
- Central defaults are in `src/config/settings.py` (e.g., `DEFAULT_OLLAMA_URL`, `DEFAULT_OLLAMA_MODEL`, `DEFAULT_LOG_LEVEL`).
- `langgraph.json` includes `"dependencies": ["./src", "./"]` so LangGraph can import graph modules.

Removed dead code/duplicates:
- Dropped `multi_agent_graph` from `watermark_remover.agent.__all__` (file was not present).
- Removed duplicate top‑level `watermark_remover/` stub that only contained `__pycache__`.

## Notes on scraping and key selection

The `scrape_music` tool has been extended beyond a stub.  Its search
strategy is now:

1. **Local search:** It looks under `data/samples` for a directory whose
   name contains the requested `title` (case insensitive).  Each
   subdirectory within that title is treated as a key or arrangement.
   If the requested `key` is present, that folder is returned.
2. **Transposition suggestions:** If the title exists but the key
   doesn’t, the tool computes direct and closest alternatives using the
   transposition helpers from `watermark_remover.utils.transposition_utils`.
   It raises a `ValueError` with a helpful message containing the
   available keys and suggested instrument/key combinations.  The
   LangChain agent can interpret this message and ask the user whether an
   alternate key or instrument would work.
3. **Online scraping:** If no title is found locally, the tool
   automatically uses Selenium to search PraiseCharts for the piece,
   click the first result, iterate through the preview images, download
   them via HTTP and save them into
   `data/samples/<sanitized_title>/<norm_key>`.  The provided
   `Dockerfile` now automatically downloads and installs the most
   recent stable Google Chrome binary and its dependencies at build
   time.  This means Selenium can run a headless Chrome browser out
   of the box without requiring you to mount a browser from the host.
   If you prefer to use a different browser (such as Chromium or
   Firefox) you can modify the Dockerfile accordingly.  Should the
   scraping logic fail for any reason (e.g. no network connectivity or
   website changes), `_scrape_with_selenium` returns `None` and
   `scrape_music` falls back to raising a `FileNotFoundError`.

If you wish to refine the scraping logic—such as selecting specific
instruments, keys or orchestration parts—you can extend
`_scrape_with_selenium` in `watermark_remover/agent/tools.py`.  The
current implementation downloads whatever preview images are available
for the first search result.  See the original
**Watermark‑Remover** repository’s `download/` and `threads/` modules for
detailed examples.

## Model weights

The repository does **not** include any trained UNet or VDSR checkpoints
to keep the project lightweight.  You can place your own `.pth` files
into `models/Watermark_Removal` and `models/VDSR` respectively.  If no
checkpoints are found the models will run with random weights, producing
nonsensical outputs but exercising the full pipeline end‑to‑end.

## Output and logging

When the agent assembles a PDF it now writes the file into the
`output` directory by default (for example, `output/output.pdf`).  Make
sure to mount this directory from your host (e.g. `-v $(pwd)/output:/app/output`)
so that the generated PDF persists after the container exits.

The agent and tools also record their actions to help you understand the
reasoning and execution order:

* **Thoughts and steps:** After each call to `run_instruction()` the
  full LLM output—including the `<think>` block and the final answer—is
  appended to `output/thoughts_and_steps.log`.  Each entry lists the
  instruction followed by the model’s reasoning and response.
* **Pipeline log:** The `wmra.tools` logger writes an execution log to
  `output/pipeline.log`.  This file records messages such as which
  directory the scraper used, which images were processed by the
  watermark removal and upscaling models, and when the PDF assembly
  completed.  The log level is controlled by the `LOG_LEVEL`
  environment variable (e.g. `DEBUG`, `INFO`).

These logs provide insight into the agent’s internal decision‑making
process and the concrete steps performed by each tool.

## Order of Worship (Parallel per‑song)

The `order-of-worship` LangGraph pipeline extracts songs from a service PDF and
now processes each song in a separate process with its own Selenium browser. This
keeps browser sessions and tool state fully isolated across songs.

- Enable/disable parallel mode: set `ORDER_PARALLEL=1` (default) or `ORDER_PARALLEL=0`.
- Control concurrency: set `ORDER_MAX_PROCS` (defaults to `min(#songs, cpu_count)`).

Each worker process sets a unique `RUN_TS` and `WMRA_LOG_DIR` so logs and temp
artifacts are kept separate, then copies assembled PDFs into `output/orders/<MM_DD_YYYY>/`
with names like `01_01_Title_Instrument_Key.pdf`.
