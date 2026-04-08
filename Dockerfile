FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend

RUN pip install --no-cache-dir --no-build-isolation ".[api]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.cmd.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
