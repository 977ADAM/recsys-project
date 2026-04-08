# Frontend

Новостной frontend лежит в этой папке и сейчас работает автономно, без `inference_service`.

Быстрый запуск:

```bash
uvicorn backend.cmd.api.main:app --reload --port 8080
```

После этого откройте:

```text
http://localhost:8080
```

Лента, рекламные слоты и карточки собираются локально в `frontend/app.js`, поэтому отдельный inference backend больше не нужен.
