# Dockerfile for the Watermark Remover Agent with full GPU and model support.
# This image is based on the official PyTorch runtime with CUDA and cuDNN.

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Install system dependencies required by PIL/opencv, reportlab and Chrome.
#
# The Selenium scraper in this project requires a working browser.  On some
# base images the `chromium` package is not available in the default
# repositories, which previously caused the build to fail.  To ensure
# consistency we download the latest stable Google Chrome binary directly
# from Google and install it via the .deb package.  This automatically
# pulls in the appropriate dependencies.  If you prefer to use a
# different browser (e.g. chromium-browser or firefox), you can modify
# the following lines accordingly.
RUN apt-get update && \
    # Install basic libraries required by PIL, OpenCV and reportlab
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        ca-certificates \
        fonts-liberation && \
    # Download and install the latest stable Google Chrome for amd64.
    # The file `google-chrome-stable_current_amd64.deb` always points to
    # the most recent stable release.  Using `apt-get install` on the
    # downloaded file ensures that all dependencies are resolved
    # automatically.
    wget -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y --no-install-recommends /tmp/chrome.deb && \
    rm /tmp/chrome.deb && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.  Torch is already provided by the base image.
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port for future use (e.g. LangGraph Studio or a web server)
EXPOSE 2024

# By default, drop into bash for interactive debugging.  You can override
# the CMD when running the container to start your own process.
CMD ["bash"]