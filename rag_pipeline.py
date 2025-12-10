# rag_pipeline.py
import time
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from pypdf import PdfReader

# HuggingFace
from transformers import pipeline, AutoTokenizer
# OpenAI
import openai


class RAGPipeline:
    def __init__(self, use_openai=False, hf_model="google/flan-t5-small"):
        """
        use_openai=True  -> uses GPT (requires API key)
        use_openai=False -> uses HuggingFace (local, free)
        hf_model -> choose "google/flan-t5-small" (fast) or "google/flan-t5-base" (slower, better)
        """
        self.vectorstore = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.use_openai = use_openai
        self.hf_model = hf_model

    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's cl100k_base tokenizer"""
        return len(self.tokenizer.encode(text))

    def load_pdf(self, pdf_file):
        """Extract text from PDF"""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    def create_vectorstore(self, text: str):
        """Split PDF into chunks, embed, and store in FAISS vector DB"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(documents, embeddings)

        total_tokens = sum(self.count_tokens(chunk) for chunk in chunks)
        return total_tokens

    def query_with_metrics(self, question: str, k: int = 3):
        """
        Query pipeline with detailed metrics:
        Returns dict with answer, tokens, time, and retrieved context
        """
        if not self.vectorstore:
            return {"error": "⚠️ Please upload a PDF first."}

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = (
            "Use the provided CONTEXT to answer the question. Do not invent new facts.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely:"
        )

        # --- OPENAI path ---
        if self.use_openai:
            start = time.monotonic()
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            end = time.monotonic()

            answer = resp["choices"][0]["message"]["content"].strip()
            usage = resp.get("usage", {})

            return {
                "answer": answer,
                "elapsed_s": round(end - start, 3),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "context": context,
                "model": "openai:gpt-3.5-turbo"
            }

        # --- HUGGINGFACE path ---
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            hf_pipeline = pipeline(
                "text2text-generation",
                model=self.hf_model,
                tokenizer=hf_tokenizer
            )

            start = time.monotonic()
            out = hf_pipeline(prompt, max_new_tokens=256, truncation=True)
            end = time.monotonic()

            answer = out[0].get("generated_text", "").strip()

            input_tokens = len(hf_tokenizer.encode(prompt))
            output_tokens = len(hf_tokenizer.encode(answer))
            total = input_tokens + output_tokens

            return {
                "answer": answer,
                "elapsed_s": round(end - start, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
                "context": context,
                "model": f"huggingface:{self.hf_model}"
            }

    def query(self, question: str):
        """Simple query (returns only answer text, for backward compatibility)"""
        res = self.query_with_metrics(question)
        if "error" in res:
            return res["error"]
        return res["answer"]