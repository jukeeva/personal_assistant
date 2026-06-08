import streamlit as st

from rag import ingest_document, answer_question

st.set_page_config(
    page_title="Smart RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Personal Knowledge Assistant")
st.write("Upload documents and ask questions about them.")

uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or Markdown file",
    type=["pdf", "txt", "md"]
)

if uploaded_file is not None:
    if st.button("Ingest document"):
        with st.spinner("Reading, chunking, embedding, and storing document..."):
            chunk_count = ingest_document(uploaded_file, uploaded_file.name)

        st.success(f"Document ingested successfully. Created {chunk_count} chunks.")

st.divider()

question = st.text_input("Ask a question about your uploaded documents:")

if question:
    with st.spinner("Searching documents and generating answer..."):
        answer = answer_question(question)

    st.subheader("Answer")
    st.write(answer)