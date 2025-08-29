"""Tests for the Watermark Remover agent and its underlying tools.

These tests exercise the core image processing functions (watermark
removal, upscaling and PDF assembly) using small dummy images to avoid
heavy computation.  They also verify that the high‑level agent entry
points return strings even when external services (e.g. Ollama) are
unavailable.

The tests intentionally avoid using pytest fixtures so they can run
with Python's built‑in ``unittest`` discovery or simple invocation of
this module.  If pytest is installed, it will still discover and run
the tests.
"""

import os
import sys
import tempfile
from typing import Iterable

# Adjust sys.path so that the `watermark_remover` package can be imported when
# running tests via `python -m unittest`.  This inserts the repository root
# (the parent of this tests directory) at the beginning of sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image

from watermark_remover.tools import (
    remove_watermark,
    upscale_images,
    assemble_pdf,
    scrape_music,
)
from watermark_remover.agent.graph_ollama import run_instruction
from watermark_remover.agent.order_of_worship_graph import _read_pdf_text

try:
    from watermark_remover.agent.ollama_agent import get_ollama_agent  # type: ignore[import-not-found]
except Exception:
    get_ollama_agent = None  # type: ignore


def _create_dummy_images(directory: str, count: int = 2) -> Iterable[str]:
    """Create a few small grayscale PNG images in ``directory``.

    Returns an iterable of file names created.  Each image is a solid
    colour of size 64×64 pixels.
    """
    filenames = []
    for i in range(count):
        img = Image.new("L", (64, 64), color=128)
        fname = f"dummy_{i}.png"
        path = os.path.join(directory, fname)
        img.save(path)
        filenames.append(fname)
    return filenames


def test_image_processing_pipeline() -> None:
    """End‑to‑end test of watermark removal, upscaling and PDF assembly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an input directory with dummy images
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        _create_dummy_images(input_dir, count=3)
        # Create empty model directories to satisfy the functions
        wmr_model = os.path.join(tmpdir, "wmr_model")
        vdsr_model = os.path.join(tmpdir, "vdsr_model")
        os.makedirs(wmr_model)
        os.makedirs(vdsr_model)
        try:
            import torch  # type: ignore
        except Exception:
            # Skip the heavy pipeline if torch is unavailable
            return
        # Run watermark removal (random weights if no checkpoints)
        processed_dir = remove_watermark.func(
            input_dir=input_dir,
            model_dir=wmr_model,
            output_dir=os.path.join(tmpdir, "processed"),
        )
        assert os.path.isdir(processed_dir)
        processed_images = [
            f for f in os.listdir(processed_dir) if f.lower().endswith(".png")
        ]
        assert len(processed_images) == 3
        # Run upscaling (random weights if no checkpoints)
        upscaled_dir = upscale_images.func(
            input_dir=processed_dir,
            model_dir=vdsr_model,
            output_dir=os.path.join(tmpdir, "upscaled"),
        )
        assert os.path.isdir(upscaled_dir)
        upscaled_images = [
            f for f in os.listdir(upscaled_dir) if f.lower().endswith(".png")
        ]
        assert len(upscaled_images) == 3
        # Assemble into a PDF
        pdf_path = assemble_pdf.func(
            image_dir=upscaled_dir, output_pdf=os.path.join(tmpdir, "output.pdf")
        )
        assert os.path.isfile(pdf_path)


def test_scrape_music_missing_piece() -> None:
    """scrape_music should raise FileNotFoundError when no piece is available."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        try:
            scrape_music.func(
                title="NoSong",
                instrument="piano",
                key="C",
                input_dir=os.path.join(repo_root, "nope"),
            )
        except FileNotFoundError:
            pass
        else:
            assert False, "Expected FileNotFoundError for missing piece"
    finally:
        os.chdir(cwd)


def test_run_instruction_returns_string() -> None:
    """run_instruction should always return a string result."""
    out = run_instruction("Say ONLY READY.")
    assert isinstance(out, str)


def test_get_ollama_agent_construction() -> None:
    """get_ollama_agent should either construct an agent or raise ImportError."""
    if get_ollama_agent is None:
        # Dependencies are missing; nothing to test
        return
    try:
        agent = get_ollama_agent(model_name="qwen3:30b")
    except ImportError:
        return
    # If construction succeeded, ensure the returned object has an invoke method
    assert hasattr(agent, "invoke")


def test_read_pdf_text(tmp_path) -> None:
    """_read_pdf_text should extract text from a simple PDF."""
    pdf_path = tmp_path / "sample.pdf"
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello world")
    c.save()

    text = _read_pdf_text(str(pdf_path))
    assert "Hello world" in text