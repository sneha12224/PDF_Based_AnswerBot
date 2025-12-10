# PDF_Based_AnswerBot
### **Objective**
PDF AnswerBot is an AI-powered Retrieval Augmented Generation (RAG) application that allows users to upload a PDF and ask questions based on its content.
The system extracts text, creates vector embeddings, stores them in FAISS, and generates accurate answers using an LLM.

## Step 1: Project Objective

The main objective of this project is to:

Read and understand PDF documents

Convert document text into semantic vectors

Retrieve relevant context for a question

Generate accurate, context-based answers

## Step 2: High-Level Workflow

User Uploads PDF
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
FAISS Vector Store
        ↓
Retriever
        ↓
LLM (HuggingFace / OpenAI)
        ↓
Answer + Metrics
        ↓
LLM (HuggingFace / OpenAI)
        ↓
Answer + Metrics

## Step 3: Technology Used

Component :	Technology
Frontend -	Streamlit
Backend -	Python
RAG Framework	- LangChain
Vector DB -	FAISS
Embeddings	- Sentence Transformers
LLM	- HuggingFace / OpenAI
PDF Parser -	PyPDF

## Step 4: Project Structure

Chatbot/
│
├── app.py               # Streamlit UI
├── rag_pipeline.py      # RAG pipeline logic
├── requirements.txt     # Project dependencies
├── venv/                # Virtual environment
└── README.md

## Step 5: How the Application Works
1️⃣ Upload PDF

The user uploads a PDF document.

2️⃣ PDF Processing

Text is extracted from the PDF

Text is split into chunks

Embeddings are generated

FAISS vector store is created

3️⃣ Ask a Question

The user enters a question related to the PDF content.

4️⃣ Answer Generation

Relevant chunks are retrieved

Context is passed to the LLM

A precise answer is generated
## Step 6: Key Modules Explained
app.py

Streamlit user interface

PDF upload handling

Question input and answer display

rag_pipeline.py

PDF text extraction

Token counting

Text chunking

FAISS vector store creation

Retrieval and LLM response

## Step 7: Use Cases

📚 Study material Q&A

🏫 Academic mini / major project

📄 Research paper analysis

🤖 AI document chatbot

🧠 Knowledge assistant
