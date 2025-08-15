FROM python:3.11

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY run.py /app/run.py

ENV TARGET_URL=http://whoop_api:8000/fetch-data
ENV RUN_INTERVAL_SECONDS=3600

CMD ["python", "-u", "run.py"]

