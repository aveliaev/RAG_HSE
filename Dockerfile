FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# torch только под CPU — иначе pip тянет CUDA-сборку на несколько гигабайт
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY bot/requirements.txt bot/requirements.txt
RUN pip install -r bot/requirements.txt

# Скачиваем модели эмбеддингов и реранкера при сборке, а не при каждом старте
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-base'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY dataset/ dataset/
COPY bot/ bot/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

WORKDIR /app/bot
ENV DATA_DIR=/data
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "bot.py"]
