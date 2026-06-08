def fixed_size_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Simple beginner chunking method.
    Splits text into character-based chunks with overlap.
    We will replace this with semantic chunking later.
    """
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks