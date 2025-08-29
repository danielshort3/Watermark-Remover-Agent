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

* **PDF text extraction**: The order-of-worship workflow reads text
  from input PDFs.  Install `pdfminer.six` (in addition to `PyPDF2`,
  which is already required) for more robust parsing:

  ```bash
  pip install pdfminer.six
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
  watermark-remover-agent
```

When the container starts you can either run an interactive REPL via
`python -m watermark_remover.agent.ollama_agent` or execute a single
instruction with the convenience function described below.

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
# tools defined in ``watermark_remover.tools``.
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

## Notes on scraping

The `scrape_music` tool relies on Selenium to fetch preview images from
PraiseCharts.  If the requested piece cannot be retrieved—for example due
to missing browser dependencies or network access—the tool raises
`FileNotFoundError`.  You can customise the scraping behaviour by
modifying `_scrape_with_selenium` in `watermark_remover/tools/scrape.py`.

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
