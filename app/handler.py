import json
from retriever import retrieve_and_answer


def handler(event, context):
    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        question = body.get("question", "").strip()
        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required field: question"})
            }

        answer = retrieve_and_answer(question)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
