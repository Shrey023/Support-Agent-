from load_documents import load_documents


def split_into_chunks(text, chunk_size=300, overlap=50):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If a paragraph itself is too long, split it into words.
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            words = paragraph.split()
            piece = ""

            for word in words:
                candidate = f"{piece} {word}".strip()

                if len(candidate) > chunk_size and piece:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = candidate

            if piece:
                current_chunk = piece

            continue

        candidate = f"{current_chunk}\n\n{paragraph}".strip()

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


if __name__ == "__main__":
    documents = load_documents("knowledge_base")

    for document in documents:
        chunks = split_into_chunks(document["content"])

        print(f"\nDocument: {document['filename']}")
        print(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {index} ---")
            print(chunk)