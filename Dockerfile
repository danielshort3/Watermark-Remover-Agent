# Dockerfile for the Watermark Remover Agent with full GPU and model support.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Copy the repository contents into the container
COPY . /app

# Install system dependencies for Pillow, OpenCV, reportlab and Chrome
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        ca-certificates \
        fonts-liberation && \
    # Download and install Google Chrome
    wget -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y --no-install-recommends /tmp/chrome.deb && \
    rm /tmp/chrome.deb && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch already provided)
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for debugging or future use
EXPOSE 2024

CMD ["bash"]