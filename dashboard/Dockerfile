FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/dashboard/requirements.txt
RUN pip install --no-cache-dir -r /app/dashboard/requirements.txt

COPY app.py /app/dashboard/app.py
COPY pages /app/dashboard/pages

WORKDIR /app/dashboard
EXPOSE 8601
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8601"] 