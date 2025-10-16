FROM python:3.8-slim

# Install system build dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install SageMaker Training Toolkit and project dependencies
RUN pip install --no-cache-dir sagemaker-training torch pandas numpy pyarrow pytest

# Copy training script into the container
COPY scripts/training/train.py /opt/ml/code/train.py
COPY scripts/training/train_lib /opt/ml/code/train_lib
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

# Use SageMaker Training Toolkit CLI for entrypoint (handles channel mounts and env)
ENV SAGEMAKER_PROGRAM=train.py
ENTRYPOINT ["python3", "-u", "-m", "sagemaker_training.cli.train"]
