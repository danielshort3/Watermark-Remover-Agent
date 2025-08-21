# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set a working directory
WORKDIR /app

# Install system dependencies needed for PyQt5 and Selenium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy source code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the LangGraph API port
EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]