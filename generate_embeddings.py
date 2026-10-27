import os
from google import genai
from dotenv import load_dotenv

from load_documents import load_documents
from chunk_documents import split_into_chunks


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file")


# Create Google AI client
client = genai.Client(api_key=api_key)


# Load our support documents
documents = load_documents("knowledge_base")

# Select an embedding model
EMBEDDING_MODEL = "gemini-embedding-001"


for document in documents:
    chunks = split_into_chunks(document["content"])

    print(f"\nDocument: {document['filename']}")

    for index, chunk in enumerate(chunks, start=1):

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk
        )

        embedding = result.embeddings[0].values

        print(f"Chunk {index}:")
        print(f"Text: {chunk[:80]}...")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")