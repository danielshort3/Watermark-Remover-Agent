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
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))
sys.path.insert(0, repo_root)

from PIL import Image

from watermark_remover.agent.tools import (
    remove_watermark,
    upscale_images,
    assemble_pdf,
    scrape_music,
    ensure_order_pdf,
)
from watermark_remover.agent.graph_ollama import run_instruction

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
    try:
        import torch  # type: ignore
    except Exception:
        # Skip the heavy pipeline if torch is unavailable
        return

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

        # Run watermark removal (random weights if no checkpoints)
        processed_dir = remove_watermark.invoke(
            {"input_dir": input_dir, "model_dir": wmr_model, "output_dir": os.path.join(tmpdir, "processed")}
        )
        assert os.path.isdir(processed_dir)
        processed_images = [f for f in os.listdir(processed_dir) if f.lower().endswith(".png")]
        assert len(processed_images) == 3
        # Run upscaling (random weights if no checkpoints)
        upscaled_dir = upscale_images.invoke(
            {"input_dir": processed_dir, "model_dir": vdsr_model, "output_dir": os.path.join(tmpdir, "upscaled")}
        )
        assert os.path.isdir(upscaled_dir)
        upscaled_images = [f for f in os.listdir(upscaled_dir) if f.lower().endswith(".png")]
        assert len(upscaled_images) == 3
        # Assemble into a PDF
        pdf_path = assemble_pdf.invoke({"image_dir": upscaled_dir, "output_pdf": os.path.join(tmpdir, "output.pdf")})
        assert os.path.isfile(pdf_path)


def test_scrape_music_uses_explicit_directory() -> None:
    """scrape_music should copy images from an explicit input_dir without network access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)
        # Include a key token in the filename so extract_key_from_filename can infer it.
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        img_name = "dummy_E_001.png"
        img_path = os.path.join(input_dir, img_name)
        img.save(img_path)

        old_cwd = os.getcwd()
        old_run_ts = os.environ.get("RUN_TS")
        os.environ["RUN_TS"] = "TEST_RUN"
        os.chdir(tmpdir)
        try:
            out_dir = scrape_music.invoke(
                {
                    "title": "FurElise",
                    "instrument": "piano",
                    "key": "C",
                    "input_dir": input_dir,
                }
            )
        finally:
            os.chdir(old_cwd)
            if old_run_ts is None:
                os.environ.pop("RUN_TS", None)
            else:
                os.environ["RUN_TS"] = old_run_ts

        assert isinstance(out_dir, str)
        assert os.path.isdir(out_dir)
        assert os.path.isfile(os.path.join(out_dir, img_name))
        # The inferred key ("E") should be reflected in the log folder structure.
        assert out_dir.endswith(os.path.join("FurElise", "unknown", "E", "piano", "1_original"))


def test_ensure_order_pdf_creates_zero_index_file() -> None:
    """ensure_order_pdf should copy the source PDF into a 00_-prefixed file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_pdf = os.path.join(tmpdir, "October 12, 2025.pdf")
        with open(source_pdf, "wb") as fh:
            fh.write(b"%PDF-1.4\n% test content\n")
        output_root = os.path.join(tmpdir, "orders")
        result = ensure_order_pdf.invoke(
            {
                "pdf_name": source_pdf,
                "order_folder": "October 12, 2025",
                "output_root": output_root,
            }
        )
        assert os.path.isfile(result)
        assert result.endswith("00_October_12_2025_Order_Of_Worship.pdf")
        # Ensure the file lives under the expected folder
        expected_dir = os.path.join(output_root, "10_12_2025")
        assert os.path.commonpath([result, expected_dir]) == expected_dir


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
        agent = get_ollama_agent(model_name="qwen3:8b")
    except ImportError:
        return
    # If construction succeeded, ensure the returned object has an invoke method
    assert hasattr(agent, "invoke")
