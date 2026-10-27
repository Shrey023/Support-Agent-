from pathlib import Path


def load_documents(folder_path):
    documents = []

    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append({
            "filename": file_path.name,
            "content": content
        })

    return documents


if __name__ == "__main__":
    documents = load_documents("knowledge_base")

    for document in documents:
        print(f"\nFile: {document['filename']}")
        print(document["content"])
