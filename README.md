# docs_pipline_311

A Python-based agent system leveraging **LangChain**, **Hugging Face** models, and vector storage.  
This pipeline comprises four main services: **MinIO**, **Qdrant**, a custom LLM API, and a custom document processing pipeline.

---

## Table of Contents
1. [Directory Structure](#directory-structure)  
2. [Setup](#setup)  
3. [Development Environment](#development-environment)  
4. [Testing](#testing)  
5. [Project Components](#project-components)  
6. [Docker Commands](#docker-commands)

---

## Directory Structure
```
[PLACEHOLDER FOR DIRECTORY STRUCTURE]
(You can specify your folders, files, and organization here once finalized)
```

---

## Setup

### Prerequisites
- **Python 3.11**
- **Docker**

### Environment Setup

1. **Create a virtual environment**:
   ```bash
   /opt/homebrew/bin/python3.11 -m venv docs_pipline_311
   ```
2. **Remove an existing environment (if necessary)**:
   ```bash
   rm -rf quantum_env_311
   ```
3. **Activate the environment**:
   ```bash
   source quantum_env_311/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install --upgrade pip
   ```

---

## Development Environment

This project uses:
- **FastAPI** for the backend API
- **LangChain** for agent orchestration
- **Hugging Face** models
- **Qdrant** for vector storage
- **PyTorch** with **Metal** support for MacOS GPU acceleration

---

## Testing

To run tests (example):
```bash
pytest
```
*(Adjust the command to match your test structure. You can have separate commands for agent, LLM API, etc.)*

---

## Project Components

1. **MinIO** – Object storage for documents  
2. **Qdrant** – Vector store for embeddings and similarity search  
3. **Custom LLM API** – FastAPI-based service exposing LLM endpoints  
4. **Custom Document Processing Pipeline** – Service for uploading and parsing documents

### Qdrant
- Default port: **6333** (HTTP) / **6334** (gRPC)  
- Web UI: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Models
- **E5-large** for embeddings  
- **DistilBERT** for text processing  
- Metal GPU acceleration support for MacOS

---

## Docker Commands

Below are useful Docker commands for running and managing containers/images in this pipeline.

### Running the Services

#### 1. MinIO
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
- `-d`: Run container in detached mode.
- `--name minio`: Name of the container.
- `-p 9000:9000`: Map MinIO’s API port.
- `-p 9001:9001`: Map MinIO’s console port.
- `-v /home/ubuntu/data:/data`: Persist data in `/home/ubuntu/data`.
- `-e MINIO_ROOT_USER="minio-student-admin"` / `-e MINIO_ROOT_PASSWORD="minio-student-admin"`: Set credentials.
- `server /data --console-address ":9001"`: Start MinIO server with console on port 9001.

#### 2. Qdrant
```bash
docker run -d \
  --name qdrant \
  --network server-net \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /home/ubuntu/qdrant:/qdrant/storage \
  -e QDRANT__SERVICE__GRPC_ENABLE=true \
  qdrant/qdrant:latest
```
- `-d`: Detached mode.
- `--name qdrant`: Container name.
- `--network server-net`: Use the same network as other services.
- `-p 6333:6333` and `-p 6334:6334`: Map Qdrant ports for HTTP and gRPC.
- `-v /home/ubuntu/qdrant:/qdrant/storage`: Persist data.
- `-e QDRANT__SERVICE__GRPC_ENABLE=true`: Enable gRPC.

#### 3. Custom LLM API

docker run -d \
  --name docs_pipeline_311 \
  -p 3009:3009 \
  docs_pipeline_311-app

#### 4. Custom Document Processing Pipeline

docker run -d \
  --name llm_api_311 \
  -p 8675:8675 \
  llm_api_311-app

---

### Stopping and Removing Containers

- **Stop the running container**:
  ```bash
  docker stop docs_pipeline_311
  docker stop llm_api_311
  ```
- **Remove the container**:
  ```bash
  docker rm docs_pipeline_311
  docker rm llm_api_311
  ```

---

### Removing Images

- **List images**:
  ```bash
  docker images
  ```
- **Remove specific images**:
  ```bash
  docker rmi [IMAGE ID]
  docker rmi [IMAGE ID]
  ```
- **Remove optional custom images**:
  ```bash
  docker rmi docs_pipeline_311-app
  docker rmi llm_api_311-app
  ```

------------------------------------------------------------------------------

### Rebuilding and Tagging Images
  
  ```
- **Build a Docker images & Tag and push your images to Docker Hub**: 
  ```bash
  docker login

-agent_sidecar_311------------------------------------------------------------------------
  docker build -t agent_sidecar_311-app .
  docker tag agent_sidecar_311-app gbeals1/api-servers:agent_sidecar_311-v1.0
  docker push gbeals1/api-servers:agent_sidecar_311-v1.0
  
-docs_pipeline_311------------------------------------------------------------------------
  docker build -t docs_pipeline_311-app .
  docker tag docs_pipeline_311-app gbeals1/api-servers:docs_pipeline_311-v1.0
  docker push gbeals1/api-servers:docs_pipeline_311-v1.0

-llm_api_311-----------------------------------------------------------------------------
  docker build -t llm_api_311-app .
  docker tag llm_api_311-app gbeals1/api-servers:llm_api_311-v1.0
  docker push gbeals1/api-servers:llm_api_311-v1.0
  ```

- **Build for multiple platforms and push**:
  ```bash
  docker buildx build --platform linux/amd64,linux/arm64 \
    -t gbeals1/api-servers:docs_pipeline_311-v1.0 --push .
------------------------------------------------------------------------------
  docker buildx build --platform linux/amd64,linux/arm64 \
    -t gbeals1/api-servers:llm_api_311-v1.0 --push .
  ```

---

### Docker Compose Commands

- **Stop and remove all containers**:
  ```bash
  docker compose down
  ```
- **Remove the Docker network**:
  ```bash
  docker network rm server-net
  ```

---

## Example API Endpoints

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

---

**Enjoy building and experimenting with your Python-based LangChain pipeline!**