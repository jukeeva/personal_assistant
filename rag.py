import os
import fitz
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from chunking import fixed_size_chunk

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="personal_knowledge"
)


def extract_text_from_pdf(file) -> str:
    """
    Extract text from an uploaded PDF file.
    """
    text = ""

    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        text += page.get_text()

    return text


def extract_text_from_txt(file) -> str:
    """
    Extract text from an uploaded TXT or Markdown file.
    """
    return file.read().decode("utf-8")


def ingest_document(file, filename: str):
    """
    Extract text, chunk it, embed chunks, and store them in ChromaDB.
    """
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".txt") or filename.endswith(".md"):
        text = extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type. Use PDF, TXT, or MD.")

    chunks = fixed_size_chunk(text)

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            "chunk_type": "fixed"
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def retrieve_context(question: str, n_results: int = 4):
    """
    Retrieve the most relevant chunks for a user question.
    """
    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def answer_question(question: str) -> str:
    """
    Retrieve context and ask the LLM to answer using only that context.
    """
    documents, metadatas = retrieve_context(question)

    context_blocks = []

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown source")
        chunk_index = meta.get("chunk_index", "Unknown chunk")

        context_blocks.append(
            f"Source: {source}, Chunk: {chunk_index}\n{doc}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a personal knowledge assistant.

Answer the user's question using only the context below.
If the answer is not in the context, say:
"I don't know based on the uploaded documents."

Always cite the source filename when possible.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You answer questions using retrieved document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content