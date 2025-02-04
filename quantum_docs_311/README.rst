============
PentestAgent
============

A Python-based agent system leveraging LangChain, Hugging Face models, and vector storage for pentest analysis.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Development Environment](#development-environment)
- [Testing](#testing)
- [Project Components](#project-components)

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

# API

- FastAPI-based REST API
- Endpoint documentation available at: http://localhost:8000/docs


