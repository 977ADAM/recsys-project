FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.cmd.api.main:app", "--host", "127.0.0.1", "--port", "8080"]
