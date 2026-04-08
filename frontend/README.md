# Frontend

Новостной frontend лежит в этой папке и по умолчанию ожидает backend proxy на `POST /api/v1/recommendations`.

Быстрый запуск:

```bash
uvicorn inference_service.main:app --reload --port 8001
uvicorn backend.cmd.api.main:app --reload --port 8080
```

После этого откройте:

```text
http://localhost:8080
```

Фронтенд по умолчанию ходит в относительный endpoint `/api/v1/recommendations`, поэтому для смены порта backend отдельная правка `frontend/app.js` не нужна.

Если inference-сервис пока не поднят, интерфейс автоматически покажет демо-ленту, чтобы можно было дорабатывать верстку и рекламные места без работающего API.
