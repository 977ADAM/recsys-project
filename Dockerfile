FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt ./
COPY backend ./backend

RUN pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "backend.cmd.api.main:app", "--host", "127.0.0.1", "--port", "8080"]
