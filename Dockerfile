FROM public.ecr.aws/lambda/python:3.11

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY app/requirements.txt .
RUN pip install -r requirements.txt

ENV HF_HOME=/opt/huggingface_cache
ENV SENTENCE_TRANSFORMERS_HOME=/opt/sentence_transformers_cache

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app/ ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
