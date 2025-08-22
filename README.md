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

## Notes on scraping

The `scrape_music` tool in this repository is still a stub: it simply
returns the contents of a directory you provide (by default
`data/samples`).  To integrate actual scraping logic (e.g. Selenium or
API calls to download sheet music) replace the body of
`scrape_music` with your implementation.  The agent's reasoning flow
will remain the same.

## Model weights

The repository does **not** include any trained UNet or VDSR checkpoints
to keep the project lightweight.  You can place your own `.pth` files
into `models/Watermark_Removal` and `models/VDSR` respectively.  If no
checkpoints are found the models will run with random weights, producing
nonsensical outputs but exercising the full pipeline end‑to‑end.