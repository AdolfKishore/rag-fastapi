
# Worksbot RAG API

A Retrieval-Augmented Generation (RAG) application built using FastAPI, LangChain, ChromaDB and Google Gemini.

## Project Overview

This project implements a document-based question-answering system.

It loads company documents, splits them into smaller chunks, generates vector embeddings, stores them in ChromaDB, retrieves relevant information, and uses a Gemini language model to generate answers.

When Gemini is unavailable due to API quota or temporary service errors, the application provides a retrieval-based fallback response.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- LangChain
- ChromaDB
- Google Gemini API
- Google Gemini Embeddings
- Pydantic
- Python Dotenv

## Project Structure

```text
RAG-B3/
│
├── data/
│   └── company.txt
│
├── chroma_db/
├── venv/
│
├── main.py
├── requirements.txt
├── .env
└── .gitignore
```

Note: The `.env`, `venv/`, and `chroma_db/` directories are excluded from GitHub.

## RAG Workflow

1. Load company documents using LangChain.
2. Split documents into smaller text chunks.
3. Generate embeddings using Google Gemini.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using similarity search.
6. Combine retrieved chunks into context.
7. Generate answers using the Gemini language model.
8. Return results through FastAPI endpoints.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API home |
| GET | `/health` | Health check |
| POST | `/retrieve` | Retrieve relevant document chunks |
| POST | `/ask` | Ask questions using RAG |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AdolfKishore/rag-fastapi.git
cd rag-fastapi
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Never upload your API key to GitHub.

### 6. Run the application

```bash
python -m uvicorn main:app --reload
```

## Access Swagger UI

Open:

http://127.0.0.1:8000/docs

## Example Request

POST `/ask`

```json
{
  "question": "What technologies are used in the multi-model chatbot project?"
}
```

POST `/retrieve`

```json
{
  "question": "Explain the RAG workflow."
}
```

## Error Handling

The application handles Gemini quota exhaustion and temporary service unavailability.

When Gemini generation cannot be completed, the API returns retrieved document information as a fallback.

## Important Note

This repository contains a starter training knowledge base. Company details and other organizational information should only be added after verification.
