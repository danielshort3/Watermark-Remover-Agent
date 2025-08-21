# Use an official PyTorch runtime as a parent image.  We base this on the
# CUDA‑enabled runtime image to ensure GPU support for both the deep learning
# models and the Ollama server if run within the container.  If you are
# targeting CPU‑only execution, you can switch to a CPU‑only variant.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set a working directory inside the container
WORKDIR /app

# Copy the source code into the container
COPY . /app

# Install system dependencies required by PyQt5, Selenium and Ollama.  We
# consolidate apt operations into a single layer to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        unzip \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install the Ollama CLI into the image.  This allows the container to
# communicate with a local or remote Ollama server and optionally run
# ``ollama serve`` itself.  See https://ollama.ai/ for installation details.
RUN curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz && \
    tar -xzf /tmp/ollama.tgz -C /usr/local && \
    chmod +x /usr/local/bin/ollama && \
    rm /tmp/ollama.tgz

# By default, Ollama looks for model blobs under $OLLAMA_MODELS.  Expose
# this environment variable so users can mount their host model cache into
# the container (e.g. ``-v $HOME/.ollama:/root/.ollama`` when running).
ENV OLLAMA_MODELS=/root/.ollama

# Install Python dependencies.  Copy requirements to a temporary location
# to leverage Docker's caching semantics.
RUN pip install --no-cache-dir -r requirements.txt

# Add the repository to the Python path.  This allows running modules
# directly via ``python -m watermark_remover ...`` without needing a
# separate installation step.
ENV PYTHONPATH=/app

# Expose the LangGraph API port.  When running ``langgraph dev``, the
# development server listens on this port by default.  You can still run
# other scripts by overriding the container command at runtime.
EXPOSE 2024

# By default, start the LangGraph development server.  To run the
# Ollama‑powered agent instead, override the command when launching the
# container (e.g. ``docker run ... python -m watermark_remover.agent.ollama_agent``).
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]