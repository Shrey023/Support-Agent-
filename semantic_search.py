import os
from google import genai
from dotenv import load_dotenv

from load_documents import load_documents
from chunk_documents import split_into_chunks

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file")

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"


def get_embedding(text):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return result.embeddings[0].values


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def build_knowledge_base():
    knowledge_base = []

    documents = load_documents("knowledge_base")

    for document in documents:
        chunks = split_into_chunks(document["content"])

        for chunk in chunks:
            embedding = get_embedding(chunk)

            knowledge_base.append({
                "filename": document["filename"],
                "text": chunk,
                "embedding": embedding
            })

    return knowledge_base


def search_knowledge_base(question, knowledge_base, top_k=2):
    question_embedding = get_embedding(question)

    scored_chunks = []

    for item in knowledge_base:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        scored_chunks.append({
            "filename": item["filename"],
            "text": item["text"],
            "score": score
        })

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


if __name__ == "__main__":
    knowledge_base = build_knowledge_base()

    question = input("\nAsk a support question: ")

    results = search_knowledge_base(
        question,
        knowledge_base
    )

    print("\nMost relevant passages:")

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"File: {result['filename']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])