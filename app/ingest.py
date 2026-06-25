import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

def ingest_document(file_path: str) -> None:
    with open(file_path, "r") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    print(f"Created {len(embeddings)} embeddings")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    vectors = [
        {
            "id": f"{os.path.basename(file_path)}-chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk, "source": file_path}
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} vectors to Pinecone")

if __name__ == "__main__":
    ingest_document("docs/sample.txt")
