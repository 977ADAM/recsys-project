# Banner Recommendation Project

Проект для персонализированной рекомендации баннеров в два этапа:
- `candidate generation` через PyTorch two-tower retrieval
- `ranking` через CatBoost CTR model

Основная активная ветка проекта:
- [app_streamlit.py](/home/adam/projects/recsys-project/app_streamlit.py)
- [main.py](/home/adam/projects/recsys-project/main.py)
- [src/pipeline/train.py](/home/adam/projects/recsys-project/src/pipeline/train.py)
- [src/pipeline/inference.py](/home/adam/projects/recsys-project/src/pipeline/inference.py)
- [src/scripts/pytorch_recsys](/home/adam/projects/recsys-project/src/scripts/pytorch_recsys)

## Что делает проект

Проект обучает и использует две модели:
- CatBoost ranker предсказывает CTR и ранжирует баннеры для пользователя
- PyTorch retrieval-модель быстро отбирает кандидатов перед ранжированием

Поддерживаются два режима рекомендаций:
- `all banners`: CatBoost скорит все баннеры
- `retrieval + ranking`: PyTorch retrieval сначала выбирает `top-N` кандидатов, потом CatBoost пересортировывает только их

## Окружение

Рекомендуется работать через локальное виртуальное окружение `.venv`.

Активация:

```bash
source .venv/bin/activate
```

Если не хочешь активировать окружение, можно запускать команды так:

```bash
./.venv/bin/python main.py -h
```

## Главная точка входа

В проекте есть единый help-entrypoint:

```bash
python main.py
```

Он показывает основные команды:
- `ui`
- `train-ranker`
- `train-retrieval`
- `recommend`

Справка:

```bash
python main.py -h
```

## Обучение CatBoost Ranker

Базовый запуск:

```bash
python main.py train-ranker
```

Эквивалент прямого запуска:

```bash
python src/pipeline/train.py \
  --interactions-csv ./data/db/banner_interactions.csv \
  --users-csv ./data/db/users.csv \
  --banners-csv ./data/db/banners.csv \
  --output-dir ctr_artifacts \
  --valid-days 14 \
  --iterations 400 \
  --learning-rate 0.05 \
  --depth 8 \
  --random-seed 42
```

Артефакты ranker-а сохраняются в папку вроде `ctr_artifacts/`.

## Обучение PyTorch Retrieval

Базовый запуск:

```bash
python main.py train-retrieval --save-item-embeddings
```

Эквивалент прямого запуска:

```bash
python src/scripts/pytorch.py \
  --epochs 5 \
  --batch-size 1024 \
  --embedding-dim 64 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --k 100 \
  --seed 42 \
  --output-dir artifacts/pytorch_retrieval \
  --save-item-embeddings
```

Что сохраняется:
- `retrieval_model.pt`
- `metadata.json`
- `item_embeddings.npy` если передан `--save-item-embeddings`

## Запуск Streamlit

Запуск через entrypoint:

```bash
python main.py ui
```

Или напрямую:

```bash
streamlit run app_streamlit.py
```

В интерфейсе можно:
- обучить CatBoost модель
- получить рекомендации для пользователя
- переключать режимы `all banners` и `retrieval + ranking`
- смотреть артефакты модели

## Inference: All Banners

Запуск ranker-а без retrieval:

```bash
python main.py recommend \
  --user-id u_00007 \
  --artifacts-dir ctr_artifacts \
  --top-k 10 \
  --exclude-seen
```

Это режим, где CatBoost скорит весь каталог баннеров.

## Inference: Retrieval + Ranking

Чтобы включить двухэтапный режим:

```bash
python main.py recommend \
  --user-id u_00007 \
  --artifacts-dir ctr_artifacts \
  --retrieval-artifacts-dir artifacts/pytorch_retrieval \
  --retrieval-top-n 100 \
  --top-k 10 \
  --exclude-seen
```

Что происходит:
1. PyTorch retrieval выбирает `top-100` кандидатов
2. CatBoost строит признаки только для этих кандидатов
3. CatBoost возвращает финальный `top-k`

## Данные

Основные входные файлы:
- `data/db/banner_interactions.csv`
- `data/db/users.csv`
- `data/db/banners.csv`

## Полезные замечания

- Системный `python` может не содержать нужные зависимости, поэтому лучше использовать `.venv`
- Если хочешь использовать режим `retrieval + ranking`, retrieval-артефакты должны быть заранее обучены и сохранены
