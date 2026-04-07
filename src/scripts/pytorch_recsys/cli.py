from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from pytorch_recsys.artifacts import save_retrieval_artifacts
from pytorch_recsys.config import parse_args
from pytorch_recsys.data import (
    BPRDataset,
    build_mappings,
    build_user_history,
    load_data,
    prepare_positive_pairs,
)
from pytorch_recsys.evaluation import evaluate_topk, print_eval
from pytorch_recsys.model import TwoTower
from pytorch_recsys.training import build_train_loader, run_epoch, set_seed


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    # 1. Загружаем interactions и режем их на train/valid/test по времени.
    train_df, valid_df, test_df = load_data()

    # 2. Переводим строковые id в индексы для Embedding-слоёв.
    user2idx, item2idx, idx2item = build_mappings()

    # 3. Оставляем только positive feedback и агрегируем дубликаты user-item.
    train_pairs = prepare_positive_pairs(train_df, user2idx, item2idx)
    valid_pairs = prepare_positive_pairs(valid_df, user2idx, item2idx)
    test_pairs = prepare_positive_pairs(test_df, user2idx, item2idx)

    train_history = build_user_history(train_pairs)
    train_dataset = BPRDataset(
        positive_pairs=train_pairs,
        user_history=train_history,
        num_items=len(item2idx),
    )
    train_loader = build_train_loader(train_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(
        n_users=len(user2idx),
        n_items=len(item2idx),
        emb_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"device: {device}")
    print(f"train positive pairs: {len(train_pairs)}")
    print(f"valid positive pairs: {len(valid_pairs)}")
    print(f"test positive pairs: {len(test_pairs)}")

    # 4. На каждой эпохе обучаем модель сравнивать positive item с sampled negative item.
    for epoch in range(1, config.epochs + 1):
        epoch_loss = run_epoch(model, train_loader, optimizer, device)
        print(f"epoch {epoch}/{config.epochs} loss: {epoch_loss:.6f}")

        valid_result = evaluate_topk(
            model=model,
            eval_pairs=valid_pairs,
            seen_history=train_history,
            num_items=len(item2idx),
            device=device,
            k=config.k,
        )
        print_eval("valid", valid_result, config.k)

    # 5. После обучения проверяем качество на test и печатаем пару примеров рекомендаций.
    test_result = evaluate_topk(
        model=model,
        eval_pairs=test_pairs,
        seen_history=train_history,
        num_items=len(item2idx),
        device=device,
        k=config.k,
    )
    print_eval("test", test_result, config.k)

    artifact_dir = save_retrieval_artifacts(
        model=model,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        embedding_dim=config.embedding_dim,
        output_dir=config.output_dir,
        save_item_embeddings=config.save_item_embeddings,
        device=device,
    )
    print(f"saved retrieval artifacts to: {artifact_dir}")

    sample_users = sorted(build_user_history(test_pairs).keys())[:3]
    if not sample_users:
        return

    model.eval()
    with torch.no_grad():
        all_items = torch.arange(len(item2idx), device=device)
        item_vectors = model.encode_item(all_items)
        print("sample predictions:")

        for user_idx in sample_users:
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            scores = torch.matmul(model.encode_user(user_tensor), item_vectors.T).squeeze(0)

            seen_items = train_history.get(user_idx, set())
            if seen_items:
                scores[torch.tensor(sorted(seen_items), dtype=torch.long, device=device)] = -torch.inf

            top_items = torch.topk(scores, k=min(config.k, len(item2idx))).indices.cpu().tolist()
            top_banner_ids = [idx2item[item_idx] for item_idx in top_items]
            print(f"  user_idx={user_idx}: {top_banner_ids}")
