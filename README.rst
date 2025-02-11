============
docs_pipline_311
============

A Python-based agent system leveraging LangChain, Hugging Face models, and vector storage for pentest analysis.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Development Environment](#development-environment)
- [Testing](#testing)
- [Project Components](#project-components)

## Useful COmmands
/opt/homebrew/bin/python3.11 -m venv docs_pipline_311
rm -rf quantum_env_311
source quantum_env_311/bin/activate
pip install -r requirements.txt
pip install --upgrade pip


## Directory Structure
================

::

docs_pipline_311/
│
├── bin/                    
├── include/                
├── lib/                   
├── src/                    
│   │
│   ├── __init__.py     
│   ├── pipline.py       # initializes pulls models from hugging face.
│   │
│   ├── models/      
│        ├── __init__.py      
│        ├── spaCy_model.py.py   # inherits the Hugging face class from initilizers.py
│        └── embedding_model.py    # inherits the Hugging face class from initilizers.py
│   
│
├── configurator.py  # for the questionary cli
├── Dockerfile
├── main.py  # delegates incoming api calls to the api folder endponits /upload  & /search
├── pyvenv.cfg
├── README.rst
├── requirements.txt
└── settings.yml   # holds the settings for the app, source of truth for app operations


## Setup

### Prerequisites
- Python 3.11+
- Conda package manager
- Docker (for Qdrant vector store)

### Environment Setup

1. Start the Qdrant vector store:
```bash
docker run -p 6333:6333 qdrant/qdrant

# Navigate to backend directory
- cd backend

# Create conda environment
- conda env create -f environment.yml

# Activate environment
- conda activate pentestagent

- pytest tests/test_vector_store/test_qdrant_connection.py -v

# Development Environment
This project uses:

# FastAPI for the backend API
- LangChain for agent orchestration
- Hugging Face models (E5-large, DistilBERT)
- Qdrant for vector storage
- PyTorch with Metal support for MacOS GPU acceleration

# Testing
Run the test suite:
- Project Components
- Vector Store

# Uses Qdrant for efficient vector storage and similarity search
- Default port: 6333
- Web UI available at: http://localhost:6333/dashboard

# Models

- E5-large for embeddings
- DistilBERT for text processing
- Metal GPU acceleration support for MacOS

4 services makup this Pipeline:
# API
- FastAPI-based REST API
- Endpoints available at: 
    - curl -X GET "http://127.0.0.1:8675/monitor
    - curl -X GET "http://127.0.0.1:8675/api/upload/health"
    - curl -X POST "http://127.0.0.1:8675/api/upload/documents" \
        -F file=@/Users/guybeals/Downloads/paper04_textual_resource.pdf
    - curl -X POST "http://127.0.0.1:8675/api/models/semantic" \ 
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is shared memory?",
           "collection_name": "document_vectors"
         }'
    - curl -X POST "http://127.0.0.1:8675/api/models/qa" \     
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is a process?",
           "collection_name": "document_vectors"
         }'
    

# MINIO
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /home/ubuntu/data:/data \
  -e MINIO_ROOT_USER="minio-student-admin" \
  -e MINIO_ROOT_PASSWORD="minio-student-admin" \
  minio/minio server /data --console-address ":9001"

-d: Run the container in detached (background) mode.
--name minio: Name the container "minio".
-p 9000:9000: Map host port 9000 to container port 9000 (for MinIO’s API).
-p 9001:9001: Map host port 9001 to container port 9001 (for MinIO’s web console).
-v /home/ubuntu/data:/data: Mount the local /home/ubuntu/data directory to /data in the container for persistent storage.
-e MINIO_ROOT_USER="minio-student-admin": Set the root username for MinIO.
-e MINIO_ROOT_PASSWORD="minio-student-admin": Set the root password for MinIO.
minio/minio: Use the official MinIO Docker image.
server /data --console-address ":9001": Start MinIO in server mode using /data as the storage directory and set the console to listen on port 9001.


#Pipeline



#Qdrant
docker run -d \
  --name qdrant \
  --network server-net \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /home/ubuntu/qdrant:/qdrant/storage \
  -e QDRANT__SERVICE__GRPC_ENABLE=true \
  qdrant/qdrant:latest

-d runs the container in detached mode.
--name qdrant sets the container’s name to qdrant.
--network server-net places the container on the same Docker network as your other services (e.g., minio).
-p 6333:6333 and -p 6334:6334 map the Qdrant default HTTP and gRPC ports to your host machine.
-v /home/ubuntu/qdrant:/qdrant/storage mounts a local directory to the container for persistent storage.
-e QDRANT__SERVICE__GRPC_ENABLE=true enables gRPC in Qdrant.



Stop the running container
- docker stop my-text-parsing-pipeline
- docker stop my-llm-api

Remove the container
- docker rm my-text-parsing-pipeline
- docker stop my-llm-api

Remove the imagefrom the VM
- docker images docker rmi [IMAGE ID] docker rmi [IMAGE ID]

Remove the image (optional, as it's an official image)
- docker rmi my-text-parsing-pipeline-app <-- supposed to be 'app' docker images // remove image from both machines docker rmi [IMAGE ID]
- docker rmi my-llm-api-app

Rebuild and tag, then cross build and push to dockerhub:
- docker build -t my-text-parsing-pipeline-app .
- docker build -t my-llm-api-app .

Tag and Push the server image to your dockerhub
- docker login 
- docker tag my-text-parsing-pipeline-app gbeals1/api-servers:text-parsing-pipeline-v1.0 docker push gbeals1/api-servers:text-parsing-pipeline-v1.0
- docker tag my-llm-api-app gbeals1/api-servers:llm-api-v1.0 docker push gbeals1/api-servers:llm-api-v1.0

build the express-server for both arm64 and amd64 - this will replace existing images pushed to dockerhub
- docker buildx build --platform linux/amd64,linux/arm64 -t gbeals1/api-servers:text-parsing-pipeline-v1.0 --push .
- docker buildx build --platform linux/amd64,linux/arm64 -t gbeals1/api-servers:llm-api-v1.0 --push .

Now pull the images on the VM using docker run commands
Stop and remove all containers created by docker-compose
- docker compose down

- docker network rm server-net

