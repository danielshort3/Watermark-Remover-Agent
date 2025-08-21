# Watermark Remover for Sheet Music

This repository contains a project aimed at removing watermarks from low‑resolution sheet music and upscaling it to high‑resolution images.  The project uses pre‑trained deep learning models (UNet and VDSR) to achieve this and includes both a graphical user interface (GUI) and a new agentic pipeline for scraping, processing, and compiling sheet music into a PDF.

## Table of Contents

- [Introduction](#introduction)
- [Pre‑trained Models](#pre‑trained-models)
- [GUI Implementation](#gui-implementation)
- [Agentic Workflow](#agentic-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Introduction

This project aims to remove watermarks from low‑resolution sheet music and upscale the images to high resolution.  The process involves using a pre‑trained UNet model to remove watermarks and a pre‑trained VDSR model to enhance the resolution.  A GUI is provided to automate the scraping, processing, and compiling of sheet music into a ready‑to‑use PDF.  In addition, a LangChain/LangGraph‑based workflow is included to demonstrate how the same pipeline can be orchestrated using agents.

## Pre‑trained Models

The repository includes the following pre‑trained models:
- **UNet** model for watermark removal
- **VDSR** model for image upscaling

These models are provided as state dictionaries and can be found in the `models/` directory.  To reduce the repository size, the weight files themselves are **not** included in this version; users should populate `models/Watermark_Removal` and `models/VDSR` with their own `.pth` files.

## GUI Implementation

A GUI built with PyQt5 is used to scrape sheet music from a specified website, run it through both the UNet and VDSR models, and compile the processed images into a PDF.  This implementation is found in the notebook `sheet_music_pyqt5.ipynb` in the original project and is unchanged in this repository.

## Agentic Workflow

The `watermark_remover/agent` package contains two agentic workflows built with **LangChain** and **LangGraph**:

* **Sequential pipeline:** This graph defines tools for scraping music, removing watermarks, upscaling images and assembling a PDF and connects them in a fixed sequence.  It is exposed under the graph ID `agent` and can be visualised in Studio or invoked via the API.  Use this when you know the step order and simply want to run the pipeline.

* **LLM‑driven pipeline:** A second graph (`agent_llm`) wraps a chat‑based agent powered by a local Ollama model.  You provide a natural‑language instruction under the `instruction` key of the graph's state (e.g. "Download sheet music for Fur Elise, remove the watermark and upscale it to a PDF").  The agent decides which tools to call and in what order to fulfil the request.  See below for details on running this graph.

To run and visualise any of the graphs locally, install the dependencies listed below and run:

```bash
pip install langgraph-cli[inmem]
langgraph dev
```

This will start a local LangGraph API server and open LangGraph Studio where you can interact with the `agent` workflow defined in `watermark_remover/agent/graph.py`.

## Installation

1. Clone the repository (or download the zip containing this code) and navigate to its root:

```bash
git clone https://github.com/danielshort3/watermark-remover.git
cd watermark-remover
```

2. Install the required packages (make sure you have `pip` and `virtualenv` installed).  The core requirements include PyTorch, torchvision, PyQt5 (for the GUI), reportlab (for PDF generation) and the LangChain/LangGraph libraries.  A sample installation command is shown below:

```bash
pip install torch torchvision
pip install PyQt5
pip install opencv-python
pip install selenium
pip install webdriver-manager
pip install reportlab
pip install langchain langgraph langchain-ollama
```

3. Ensure the pre‑trained model weights are placed in the `models/` directory.  For example, the latest UNet checkpoint should be placed in `models/Watermark_Removal/` and the latest VDSR checkpoint in `models/VDSR/`.

## Usage

There are three primary ways to use this project:

1. **GUI**: Launch the GUI by running the `sheet_music_pyqt5.ipynb` notebook.  Use the GUI to scrape sheet music from a specified website, run it through both the UNet and VDSR models, and compile the processed images into a PDF.  This interface remains unchanged from the original project.

2. **Sequential pipeline via LangGraph (`agent`)**: Run `langgraph dev` in the repository root to start a local API server and open LangGraph Studio.  In Studio, select the `agent` assistant and provide a state with keys `title`, `instrument`, `key` and (optionally) `input_dir` to trigger the pipeline.  The tools defined in `watermark_remover/agent/tools.py` will execute sequentially, producing a PDF at the end.

3. **LLM‑driven pipeline via LangGraph (`agent_llm`)**: If you have [Ollama](https://ollama.ai/) installed and have pulled a large language model such as `qwen3:30b`, you can execute the entire pipeline by writing a single sentence.  Start the Ollama server (`ollama serve`) and pull the model (`ollama pull qwen3:30b`).  Then start the LangGraph API server with `langgraph dev` and, in Studio, select the graph ID `agent_llm`.  Provide a JSON payload like:

```json
{
  "instruction": "Download sheet music for Fur Elise, remove the watermark, upscale it and save it as a PDF"
}
```

The agent will parse your instruction and call the necessary tools automatically.  The final answer (success message or error) will be returned under the `result` key.

4. **Ollama‑powered chat agent (CLI)**: You can also run the chat agent outside of LangGraph.  Start the Ollama server and pull your model, then run the interactive agent script:

```bash
python -m watermark_remover.agent.ollama_agent --model qwen3:30b
```

The script will connect to your local Ollama server, load the specified model and launch a simple REPL where you can type commands like "Remove watermarks from the images in `data/samples` and make a PDF."  The agent uses the model's reasoning capabilities to decide which tools to call and in what order, and will report back with the results.

## Project Structure

The repository's code is organised into a Python package located in `watermark_remover/`:

- `gui/` – application window and PyQt dialogs (from the original project)
- `download/` – Selenium helpers and batch processing logic (from the original project)
- `threads/` – background worker threads (from the original project)
- `inference/` – model definitions and loading utilities (unchanged)
- `agent/` – **new** tools and graph definition for the LangGraph workflow
- `utils/` – shared utility functions such as transposition helpers

Model‑training notebooks remain at the repository root and are unchanged.