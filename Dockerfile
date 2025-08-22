# Dockerfile for the Watermark Remover Agent with full GPU and model support.
# This image is based on the official PyTorch runtime with CUDA and cuDNN.

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Install system dependencies required by PIL/opencv and reportlab
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.  Torch is already provided by the base image.
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port for future use (e.g. LangGraph Studio or a web server)
EXPOSE 2024

# By default, drop into bash for interactive debugging.  You can override
# the CMD when running the container to start your own process.
CMD ["bash"]