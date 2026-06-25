import os
import json
from contextlib import nullcontext
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_URI:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("p1-doc-assistant")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "bedrock"

_embedding_model = None
_pinecone_index = None
_anthropic_client = None
_bedrock_client = None


def _call_anthropic(prompt: str) -> tuple[str, int, int]:
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = _anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return (
        message.content[0].text,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )


def _call_bedrock(prompt: str) -> tuple[str, int, int]:
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = _bedrock_client.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    result = json.loads(response["body"].read())
    return (
        result["content"][0]["text"],
        result["usage"]["input_tokens"],
        result["usage"]["output_tokens"],
    )


def retrieve_and_answer(question: str, top_k: int = 3) -> str:
    global _embedding_model, _pinecone_index

    run_ctx = mlflow.start_run() if MLFLOW_URI else nullcontext()

    with run_ctx:
        if MLFLOW_URI:
            mlflow.log_param("question", question)
            mlflow.log_param("llm_provider", LLM_PROVIDER)
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

        prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say "
            "'I don't know based on the provided documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        if LLM_PROVIDER == "bedrock":
            answer, input_tokens, output_tokens = _call_bedrock(prompt)
        else:
            answer, input_tokens, output_tokens = _call_anthropic(prompt)

        if MLFLOW_URI:
            mlflow.log_metric("input_tokens", input_tokens)
            mlflow.log_metric("output_tokens", output_tokens)
            mlflow.set_tag("answer", answer[:250])

    return answer


if __name__ == "__main__":
    print("--- Query 1 ---")
    print(retrieve_and_answer("What is the remote work policy?"))
    print("\n--- Query 2 ---")
    print(retrieve_and_answer("What are the vacation days?"))
