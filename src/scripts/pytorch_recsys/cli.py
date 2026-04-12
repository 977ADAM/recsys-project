from __future__ import annotations

import math

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
    build_hard_negative_pools,
    build_mappings,
    build_user_history,
    load_data,
    prepare_positive_pairs,
)
from pytorch_recsys.evaluation import (
    evaluate_topk,
    print_eval,
    print_eval_cold_start,
    split_eval_pairs,
)
from pytorch_recsys.features import build_banner_feature_matrix, build_user_feature_matrix
from pytorch_recsys.model import TwoTower
from pytorch_recsys.training import build_train_loader, run_epoch, set_seed


def evaluate_and_print(
    split_name: str,
    model: TwoTower,
    eval_pairs,
    train_history: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    k: int,
):
    eval_split = split_eval_pairs(eval_pairs, set(train_history))
    print_eval_cold_start(split_name, eval_split)
    result = evaluate_topk(
        model=model,
        eval_pairs=eval_split.warm_pairs,
        seen_history=train_history,
        num_items=num_items,
        device=device,
        k=k,
    )
    print_eval(split_name, result, k)
    return result


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    # 1. Загружаем interactions и режем их на train/valid/test по времени.
    train_df, valid_df, test_df = load_data()

    # 2. Переводим строковые id в индексы для Embedding-слоёв.
    user2idx, item2idx, idx2item = build_mappings(train_df)
    ordered_user_ids = [user_id for user_id, _ in sorted(user2idx.items(), key=lambda pair: pair[1])]
    ordered_item_ids = [item_id for item_id, _ in sorted(item2idx.items(), key=lambda pair: pair[1])]
    user_feature_table = torch.from_numpy(build_user_feature_matrix(ordered_user_ids))
    item_feature_table = torch.from_numpy(build_banner_feature_matrix(ordered_item_ids))

    # 3. Оставляем только positive feedback и агрегируем дубликаты user-item.
    train_pairs = prepare_positive_pairs(train_df, user2idx, item2idx)
    valid_pairs = prepare_positive_pairs(valid_df, user2idx, item2idx)
    test_pairs = prepare_positive_pairs(test_df, user2idx, item2idx)

    train_history = build_user_history(train_pairs)
    hard_negative_pools = build_hard_negative_pools(train_df, user2idx, item2idx)
    train_dataset = BPRDataset(
        positive_pairs=train_pairs,
        user_history=train_history,
        num_items=len(item2idx),
        hard_negative_pools=hard_negative_pools,
    )
    train_loader = build_train_loader(train_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(
        n_users=len(user2idx),
        n_items=len(item2idx),
        emb_dim=config.embedding_dim,
        user_feature_dim=user_feature_table.shape[1],
        item_feature_dim=item_feature_table.shape[1],
        user_feature_table=user_feature_table,
        item_feature_table=item_feature_table,
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
    print(f"evaluating top-{config.k} candidates")

    best_metric_name = f"valid_recall@{config.k}"
    best_metric_value = -math.inf
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    # 4. На каждой эпохе обучаем модель сравнивать positive item с sampled negative item.
    for epoch in range(1, config.epochs + 1):
        epoch_loss = run_epoch(model, train_loader, optimizer, device)
        print(f"epoch {epoch}/{config.epochs} loss: {epoch_loss:.6f}")

        valid_result = evaluate_and_print(
            split_name="valid",
            model=model,
            eval_pairs=valid_pairs,
            train_history=train_history,
            num_items=len(item2idx),
            device=device,
            k=config.k,
        )
        current_metric = valid_result.recall_at_k
        improvement = current_metric - best_metric_value

        if improvement > config.early_stopping_min_delta:
            best_metric_value = current_metric
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            print(
                f"best epoch updated: {best_epoch} "
                f"({best_metric_name}={best_metric_value:.6f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"no improvement for {epochs_without_improvement} epoch(s); "
                f"best {best_metric_name}={best_metric_value:.6f} at epoch {best_epoch}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"early stopping triggered after epoch {epoch}; "
                f"restoring best epoch {best_epoch}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 5. После обучения проверяем качество на test и печатаем пару примеров рекомендаций.
    evaluate_and_print(
        split_name="test",
        model=model,
        eval_pairs=test_pairs,
        train_history=train_history,
        num_items=len(item2idx),
        device=device,
        k=config.k,
    )

    artifact_dir = save_retrieval_artifacts(
        model=model,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        embedding_dim=config.embedding_dim,
        output_dir=config.output_dir,
        save_item_embeddings=config.save_item_embeddings,
        device=device,
        best_epoch=best_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
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
