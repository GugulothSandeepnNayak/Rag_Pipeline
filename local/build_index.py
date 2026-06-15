import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model (768-dim)
model = SentenceTransformer("all-mpnet-base-v2")

# Load your chunks
with open("index/chunks.json", "r") as f:
    chunks = json.load(f)

# Generate embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]  # should be 768
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "index/faiss.index")

print("✅ Index built successfully with dimension:", dimension)