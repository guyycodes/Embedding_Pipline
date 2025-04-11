# Embedding_pipeline_311

## ONLY Use this readme if you want to setup the whole API
## If you want to test the pipline only, view the readme.md inside the docs_pipline_311 file

A Python-based agent system leveraging **LangChain**, **Hugging Face** models, and vector storage.  
This pipeline comprises four main services: **MinIO**, **Qdrant**, a custom LLM API, and a custom document processing pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Setup](#setup)
   - [Prerequisites](#prerequisites)
   - [Environment Setup](#environment-setup)
   - [Creating and Activating the Virtual Environment](#creating-and-activating-the-virtual-environment)
   - [Installing Dependencies](#installing-dependencies)
4. [Development Environment](#development-environment)
5. [Project Components](#project-components)
6. [Docker Container Setup](#docker-container-setup)
   - [Building Images](#building-images)
   - [Running Services](#running-services)
   - [Container Management](#container-management)
   - [Docker with GPU Access](#docker-with-gpu-access)
7. [Testing](#testing)
8. [API Endpoints](#api-endpoints)
9. [Troubleshooting and Additional Notes](#troubleshooting-and-additional-notes)

## Overview

This project provides a comprehensive document processing pipeline with LLM capabilities. It uses:
- **FastAPI** for the backend API
- **LangChain** for agent orchestration
- **Hugging Face** models
- **Qdrant** for vector storage
- **PyTorch** with **Metal** support for MacOS GPU acceleration

## Directory Structure
```
[PLACEHOLDER FOR DIRECTORY STRUCTURE]
(You can specify your folders, files, and organization here once finalized)
```

## Setup

### Prerequisites
- **Python 3.11** (or later)
- **Docker**
- A working terminal or command prompt

### Environment Setup

There are two approaches to setting up the environment. Choose the one that fits your workflow:

#### Creating and Activating the Virtual Environment

**Option 1:**
1. **Navigate to the project directory:**
   ```bash
   cd ~/De/G/Embedding_Pipline/docs_pipeline_311
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv pipeline_311
   ```

3. **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source pipeline_311/bin/activate
   ```
   
   On Windows:
   ```bash
   pipeline_311\Scripts\activate
   ```

### Installing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Development Environment

This project leverages several technologies:
- **FastAPI** for the backend API
- **LangChain** for agent orchestration
- **Hugging Face** models
- **Qdrant** for vector storage
- **PyTorch** with **Metal** support for MacOS GPU acceleration

## Project Components

2. **Qdrant** – Vector store for embeddings and similarity search  
3. **Custom LLM API** – FastAPI-based service exposing LLM endpoints  
4. **Custom Document Processing Pipeline** – Service for uploading and parsing documents

### Qdrant
- Default port: **6333** (HTTP) / **6334** (gRPC)  
- Web UI: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Models
- **E5-large** for embeddings  
- Metal GPU acceleration support for MacOS

## Docker Container Setup

### Building Images

From your project directory (where your Dockerfile resides):

#### Building the Main App
```bash
docker build -t docs_pipeline_311-app .
```

#### Building the API
```bash
docker build -t llm_api_311-app .
```

#### Building the Agent Sidecar (for running an agent that controls a virtual envirionment via x11VNC connection)
```bash
docker build -t agent_sidecar_311-app .
```

### Optional: Tag & Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Main app
docker tag docs_pipeline_311-app gbeals1/api-servers:docs_pipeline_311-v1.0
docker push gbeals1/api-servers:docs_pipeline_311-v1.0

# API
docker tag llm_api_311-app gbeals1/api-servers:llm_api_311-v1.0
docker push gbeals1/api-servers:llm_api_311-v1.0

# Agent sidecar
docker tag agent_sidecar_311-app gbeals1/api-servers:agent_sidecar_311-v1.0
docker push gbeals1/api-servers:agent_sidecar_311-v1.0
```

#### Build for Multiple Platforms and Push
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t gbeals1/api-servers:docs_pipeline_311-v1.0 --push .

docker buildx build --platform linux/amd64,linux/arm64 \
  -t gbeals1/api-servers:llm_api_311-v1.0 --push .
```

### Running Services

#### 1. MinIO (THIS IS OPTIONAL)
```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /home/ubuntu/data:/data \
  -e MINIO_ROOT_USER="minio-student-admin" \
  -e MINIO_ROOT_PASSWORD="minio-student-admin" \
  minio/minio server /data --console-address ":9001"
```

#### 2. Qdrant
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -e QDRANT__SERVICE__GRPC_ENABLE=true \
  qdrant/qdrant:latest
```

#### 3. Custom Document Processing Pipeline
```bash
docker run -d \
  --name docs_pipeline_311 \
  -p 3009:3009 \
  docs_pipeline_311-app
```

#### 4. Custom LLM API
```bash
docker run -d \
  --name llm_api_311 \
  -p 8675:8675 \
  llm_api_311-app
```

### Container Management

#### Viewing Logs
```bash
docker logs -f llm_api_311
```

#### Stopping and Removing Containers
```bash
# Stop containers
docker stop docs_pipeline_311
docker stop llm_api_311

# Remove containers
docker rm docs_pipeline_311
docker rm llm_api_311
```

#### Managing Images
```bash
# List images
docker images

# Remove specific images
docker rmi [IMAGE ID]

# Remove custom images
docker rmi docs_pipeline_311-app
docker rmi llm_api_311-app
```

#### Docker Compose Commands
```bash
# Stop and remove all containers
docker compose down

# Remove the Docker network
docker network rm server-net
```

### Docker with GPU Access

#### Overview
To leverage GPU acceleration in a cloud environment (since local machines like a MacBook may not support MPS for GPU workloads), run your application inside a Docker container on a GPU-enabled cloud instance.

#### Provisioning a GPU-Enabled Cloud Instance
Choose a cloud provider (AWS, Google Cloud, Azure) that offers GPU-enabled instances (e.g., AWS P3, Google Cloud with NVIDIA Tesla GPUs, Azure NC series).

Launch an instance with the required GPU hardware and secure SSH access.

#### Installing NVIDIA Drivers and the NVIDIA Container Toolkit

On your GPU-enabled instance (assuming an Ubuntu-based OS):

1. **Install NVIDIA Drivers:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential dkms
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-525
   nvidia-smi   # Verify that the drivers are installed
   ```

2. **Install Docker (if not already installed):**
   ```bash
   sudo apt-get install -y docker.io
   sudo systemctl enable --now docker
   ```

3. **Install the NVIDIA Docker Container Toolkit:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. **Run the container with GPU access:**

   Use the `--gpus all` flag to expose the host GPUs to the container:
   ```bash
   docker run --gpus all -p 3009:3009 docs_pipeline_311-app
   ```

#### Verifying GPU Access

After starting the container, verify GPU access by opening a shell in the container and running:

```bash
docker exec -it <container_id> /bin/bash
nvidia-smi
```

## Testing

To run tests:
```bash
pytest
```
*(Adjust the command to match your test structure. You can have separate commands for agent, LLM API, etc.)*

## API Endpoints

**(Local usage with FastAPI-based services)**

1. **Health checks**:  
   ```bash
   curl -X GET "http://127.0.0.1:8675/monitor"
   curl -X GET "http://127.0.0.1:8675/api/upload/health"
   ```

2. **Document upload**:  
   ```bash
   curl -X POST "http://127.0.0.1:8675/api/upload/documents" \
        -F file=@/path/to/your-file.pdf
   ```

3. **Semantic search**:  
   ```bash
   curl -X POST "http://127.0.0.1:8675/api/models/semantic" \
        -H "Content-Type: application/json" \
        -d '{
              "query": "What is shared memory?",
              "collection_name": "document_vectors"
            }'
   ```

4. **Question-Answering**:  
   ```bash
   curl -X POST "http://127.0.0.1:8675/api/models/qa" \
        -H "Content-Type: application/json" \
        -d '{
              "query": "What is a process?",
              "collection_name": "document_vectors"
            }'
   ```

*(Adjust port numbers and endpoint paths as needed for your setup.)*

## Troubleshooting and Additional Notes

- When working with GPUs, ensure all drivers are properly installed and compatible with your container setup.
- For MacOS users, Metal support is enabled for PyTorch, providing GPU acceleration.
- If experiencing network issues with Docker containers, verify the `server-net` network is properly configured.
- Check logs for detailed error messages when troubleshooting container issues.

**Enjoy building and experimenting with your Python-based LangChain pipeline!**