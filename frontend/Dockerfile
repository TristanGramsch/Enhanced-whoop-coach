FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY app.py /app/frontend/app.py
COPY pages /app/frontend/pages

WORKDIR /app/frontend
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]