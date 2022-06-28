FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip3 install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir -p /app/src/ && \
    mkdir -p /app/model && \
    mkdir -p /app/data && \
	mkdir -p /app/outputs \
	mkdir -p /app/transformers_cache \
	mkdir -p /app/lightning_logs

COPY train.py /app/src/

ENV DATA_SOURCE=/app/data
ENV OUTPUT_PATH=/app/outputs
ENV TRANSFORMERS_CACHE_PATH=/app/transformers_cache
ENV LIGHTNING_LOGS_PATH=/app/lightning_logs
ENV NUM_GPUS=1


RUN chgrp -R 65534 /app && \
    chmod -R 770 /app
