import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

model = SentenceTransformer("all-mpnet-base-v2")

# Use a persistent directory so different processes share the same DB
client = chromadb.Client(Settings(is_persistent=True, persist_directory="./chroma_db", allow_reset=True))
collection = client.get_or_create_collection(name="rag_docs")


def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    else:
        return open(path, "r", encoding="utf-8").read()


def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks


def ingest_documents(folder="data"):
    files = os.listdir(folder)

    all_chunks = []
    ids = []

    for file in files:
        path = os.path.join(folder, file)
        text = read_file(path)

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            ids.append(f"{file}_{i}")

    # Add in smaller batches to avoid large memory spikes / segfaults
    batch_size = 32
    total = len(all_chunks)
    stored = 0

    for i in range(0, total, batch_size):
        batch_docs = all_chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        try:
            embeddings = model.encode(batch_docs, convert_to_numpy=True).tolist()

            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                ids=batch_ids
            )

            stored += len(batch_docs)
            print(f"✅ Stored {stored}/{total} chunks in Chroma")
        except Exception as e:
            print(f"⚠️ Error storing batch {i}-{i+len(batch_docs)}: {e}")
            # continue with next batch
            continue


if __name__ == "__main__":
    ingest_documents()