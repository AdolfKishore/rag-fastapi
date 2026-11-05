
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "company.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "worksbot_knowledge"

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)


# ==================================================
# 2. DOCUMENT LOADING
# ==================================================

def load_documents() -> list[Document]:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Document not found: {DATA_FILE}"
        )

    loader = TextLoader(
        str(DATA_FILE),
        encoding="utf-8"
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No documents were loaded.")

    print(f"Documents loaded: {len(documents)}")

    return documents


# ==================================================
# 3. DOCUMENT SPLITTING
# ==================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# ==================================================
# 4. DOCUMENT RETRIEVAL
# ==================================================

def retrieve_documents(vectorstore, question, k=3):

    results = vectorstore.similarity_search_with_score(
        query=question,
        k=k
    )

    return results


# ==================================================
# 5. CONTEXT CREATION
# ==================================================

def create_context(documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# ==================================================
# 6. GEMINI ANSWER GENERATION
# ==================================================

def generate_answer(llm, question, context):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a company information assistant.

            Answer the user's question using only the
            information provided in the context.

            If the answer is not available in the context,
            respond with:

            I couldn't find that information in the
            provided company document.

            Do not invent facts.
            Do not make unsupported assumptions.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Context:
            {context}
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question": question,
        "context": context
    })

    return answer


# ==================================================
# 7. FASTAPI APPLICATION INITIALIZATION
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Worksbot RAG application...")

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    # Load company documents
    documents = load_documents()

    # Split documents into chunks
    chunks = split_documents(documents)

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    # Initialize ChromaDB
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Check existing vector database records
    existing_records = vectorstore.get(limit=1)

    if not existing_records.get("ids"):

        print("Generating document embeddings...")

        vectorstore.add_documents(
            documents=chunks
        )

        print("Documents stored successfully.")

    else:

        print("Existing ChromaDB collection loaded.")

    # Initialize Gemini language model
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )

    # Store initialized components
    app.state.vectorstore = vectorstore
    app.state.llm = llm

    print("Worksbot RAG application is ready.")

    yield

    print("Shutting down Worksbot RAG application.")


# ==================================================
# 8. CREATE FASTAPI APP
# ==================================================

app = FastAPI(
    title="Worksbot RAG API",
    description=(
        "Retrieval Augmented Generation using "
        "FastAPI, LangChain, ChromaDB and Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==================================================
# 9. REQUEST VALIDATION
# ==================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Enter your question"
    )


# ==================================================
# 10. HOME ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Worksbot RAG API is running"
    }


# ==================================================
# 11. HEALTH CHECK ENDPOINT
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "success",
        "message": "RAG API is healthy"
    }


# ==================================================
# 12. RETRIEVAL-ONLY ENDPOINT
# ==================================================

@app.post("/retrieve")
def retrieve_only(
    payload: QuestionRequest,
    http_request: Request
):

    try:

        question = payload.question.strip()

        vectorstore = http_request.app.state.vectorstore

        retrieved_results = retrieve_documents(
            vectorstore,
            question,
            k=3
        )

        results = []

        for document, score in retrieved_results:

            results.append({
                "content": document.page_content,
                "source": document.metadata.get(
                    "source", "unknown"
                ),
                "distance_score": round(
                    float(score), 4
                )
            })

        return {
            "status": "success",
            "question": question,
            "retrieved_chunks": len(results),
            "retrieved_documents": results
        }

    except Exception as error:

        print(f"Retrieval Error: {repr(error)}")

        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed."
        )


# ==================================================
# 13. MAIN RAG QUESTION-ANSWERING ENDPOINT
# ==================================================

@app.post("/ask")
def ask_question(
    payload: QuestionRequest,
    http_request: Request
):

    try:

        # Get user question
        question = payload.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Access initialized components
        vectorstore = http_request.app.state.vectorstore
        llm = http_request.app.state.llm

        # Retrieve relevant documents
        retrieved_results = retrieve_documents(
            vectorstore,
            question,
            k=3
        )

        # Handle empty retrieval
        if not retrieved_results:

            return {
                "status": "success",
                "question": question,
                "answer": (
                    "No documents were retrieved "
                    "for this question."
                ),
                "generation_mode": "no_retrieval",
                "retrieved_chunks": 0,
                "sources": [],
                "context": ""
            }

        # Extract documents from retrieval tuples
        documents = [
            document
            for document, score in retrieved_results
        ]

        # Create context
        context = create_context(documents)

        # Extract source filenames
        sources = list({
            Path(
                str(document.metadata.get("source", "unknown"))
            ).name
            for document in documents
        })

        # Default response mode
        generation_mode = "gemini"

        # Generate answer using Gemini
        try:

            answer = generate_answer(
                llm,
                question,
                context
            )

        except Exception as model_error:

            error_message = str(model_error).upper()

            quota_error = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            service_error = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            )

            if quota_error:

                print(
                    "Gemini quota exhausted. "
                    "Using retrieval fallback."
                )

            elif service_error:

                print(
                    "Gemini temporarily unavailable. "
                    "Using retrieval fallback."
                )

            else:

                print(
                    f"Gemini generation failed: "
                    f"{repr(model_error)}"
                )

            # Return retrieved information instead
            answer = (
                "AI answer generation is currently unavailable. "
                "The following information was retrieved "
                "from the company document:\n\n"
                + context
            )

            generation_mode = "retrieval_fallback"

        # Final response
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "generation_mode": generation_mode,
            "retrieved_chunks": len(documents),
            "sources": sources,
            "context": context
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"RAG Error: {repr(error)}")

        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(error)}"
        )
