import os
from contextlib import nullcontext
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_URI:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("p1-doc-assistant")

_embedding_model = None
_pinecone_index = None
_anthropic_client = None


def retrieve_and_answer(question: str, top_k: int = 3) -> str:
    global _embedding_model, _pinecone_index, _anthropic_client
    model_name = "claude-haiku-4-5-20251001"

    run_ctx = mlflow.start_run() if MLFLOW_URI else nullcontext()

    with run_ctx:
        if MLFLOW_URI:
            mlflow.log_param("question", question)
            mlflow.log_param("model", model_name)
            mlflow.log_param("top_k", top_k)

        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        question_vector = _embedding_model.encode(question).tolist()

        if _pinecone_index is None:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            _pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        results = _pinecone_index.query(
            vector=question_vector,
            top_k=top_k,
            include_metadata=True
        )

        chunks = [match["metadata"]["text"] for match in results["matches"]]
        context = "\n\n---\n\n".join(chunks)

        if MLFLOW_URI:
            mlflow.log_metric("chunks_retrieved", len(chunks))

        if _anthropic_client is None:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say "
            "'I don't know based on the provided documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        message = _anthropic_client.messages.create(
            model=model_name,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = message.content[0].text

        if MLFLOW_URI:
            mlflow.log_metric("input_tokens", message.usage.input_tokens)
            mlflow.log_metric("output_tokens", message.usage.output_tokens)
            mlflow.set_tag("answer", answer[:250])

    return answer


if __name__ == "__main__":
    print("--- Query 1 ---")
    print(retrieve_and_answer("What is the remote work policy?"))
    print("\n--- Query 2 ---")
    print(retrieve_and_answer("What are the vacation days?"))
