FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.26.4 \
    pyarrow==15.0.2 \
    boto3==1.35.21 \
    torch==2.5.1

COPY scripts/training/train_lib /opt/ml/code/train_lib
COPY scripts/inference/inference.py /opt/ml/code/inference.py
COPY hf_eolus /opt/ml/code/hf_eolus

# Preserve compatibility with legacy flat config paths
RUN set -eux; \
    mkdir -p /opt/ml/code/config; \
    find /opt/ml/code/hf_eolus -type f \
        \( -name '*.json' -o -name '*.yml' -o -name '*.yaml' \) \
        -path '*/config/*' -print0 | \
    while IFS= read -r -d '' file; do \
        ln -sf "$file" "/opt/ml/code/config/$(basename "$file")"; \
    done

ENV PYTHONPATH="/opt/ml/code"
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
