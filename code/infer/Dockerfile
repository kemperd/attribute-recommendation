FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip3 install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir -p /app/src

RUN chgrp -R nogroup /app && \
    chmod -R 770 /app

COPY infer.py /app/src/

ENV MODEL_PATH=/mnt/models
ENV NUM_GPUS=1

CMD ["gunicorn", "--chdir", "/app/src", "infer:app", "-b", "0.0.0.0:9001"]
