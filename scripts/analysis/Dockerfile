FROM python:3.11-slim

# Install core scientific stack and CPU-only PyTorch
RUN python -V && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        pyarrow==16.1.0 \
        matplotlib==3.9.2 && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /work

