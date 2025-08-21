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

The `watermark_remover/agent` package contains a proof‑of‑concept multi‑agent workflow built with **LangChain** and **LangGraph**.  This workflow defines tools for scraping music, removing watermarks, upscaling images and assembling a PDF.  A `StateGraph` orchestrates these tools into a linear pipeline.  To run and visualise the workflow locally, install the dependencies listed below and run:

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

There are two ways to use this project:

1. **GUI**: Launch the GUI by running the `sheet_music_pyqt5.ipynb` notebook.  Use the GUI to scrape sheet music from a specified website, run it through both the UNet and VDSR models, and compile the processed images into a PDF.  This interface remains unchanged from the original project.

2. **Agentic pipeline**: Run `langgraph dev` in the repository root to start a local API server and open LangGraph Studio.  In Studio, select the `agent` assistant and provide a state with keys `title`, `instrument` and `key` to trigger the pipeline.  The tools defined in `watermark_remover/agent/tools.py` will execute sequentially, producing a PDF at the end.

## Project Structure

The repository's code is organised into a Python package located in `watermark_remover/`:

- `gui/` – application window and PyQt dialogs (from the original project)
- `download/` – Selenium helpers and batch processing logic (from the original project)
- `threads/` – background worker threads (from the original project)
- `inference/` – model definitions and loading utilities (unchanged)
- `agent/` – **new** tools and graph definition for the LangGraph workflow
- `utils/` – shared utility functions such as transposition helpers

Model‑training notebooks remain at the repository root and are unchanged.
