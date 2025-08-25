FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/whoop_client/requirements.txt
RUN pip install --no-cache-dir -r /app/whoop_client/requirements.txt

COPY whoop_server.py /app/whoop_client/whoop_server.py
COPY whoop_data_fetcher.py /app/whoop_client/whoop_data_fetcher.py
COPY token.json /app/whoop_client/token.json

WORKDIR /app/whoop_client
EXPOSE 8000
CMD ["python", "whoop_server.py"]