"""训练一个代表性有限代数乘法模型。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config
from algebra import cyclic_group_algebra_table
from model import ElementTransformer


class MultiplicationDataset(Dataset):
    """将乘法表中的 pair id 转换为三 token 输入。"""

    def __init__(self, pair_ids, table):
        self.pair_ids = np.asarray(pair_ids, dtype=np.int64)
        self.table = table
        self.ring_size = len(table)

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, index):
        pair_id = int(self.pair_ids[index])
        left = pair_id // self.ring_size
        right = pair_id % self.ring_size
        tokens = torch.tensor(
            [left, self.ring_size, right], dtype=torch.long
        )
        target = torch.tensor(int(self.table[left, right]), dtype=torch.long)
        return tokens, target


def select_device(requested: str):
    """选择 CUDA、MPS 或 CPU。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=2048):
    """返回完整数据集的 loss 与 accuracy。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for tokens, targets in DataLoader(dataset, batch_size=batch_size):
        tokens = tokens.to(device)
        targets = targets.to(device)
        logits = model(tokens)
        total_loss += F.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        total_correct += int((logits.argmax(-1) == targets).sum())
        total += len(targets)
    model.train()
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algebra", choices=config.ALGEBRAS, default="F3_Z4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=config.TOTAL_STEPS)
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    algebra_config = config.ALGEBRAS[args.algebra]
    ring_size = algebra_config["p"] ** algebra_config["group_order"]
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.output_root / f"{args.algebra}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table = cyclic_group_algebra_table(
        algebra_config["p"], algebra_config["group_order"]
    )
    np.save(run_dir / "multiplication_table.npy", table)

    rng = np.random.RandomState(args.seed)
    pair_ids = rng.permutation(ring_size * ring_size)
    n_train = int(len(pair_ids) * config.TRAIN_FRACTION)
    train_dataset = MultiplicationDataset(pair_ids[:n_train], table)
    test_dataset = MultiplicationDataset(pair_ids[n_train:], table)
    loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = ElementTransformer(
        ring_size=ring_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_mlp=config.D_MLP,
        n_layers=config.N_LAYERS,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )

    metadata = {
        "algebra": args.algebra,
        "seed": args.seed,
        "p": algebra_config["p"],
        "group_order": algebra_config["group_order"],
        "ring_size": ring_size,
        "d_model": config.D_MODEL,
        "n_heads": config.N_HEADS,
        "d_mlp": config.D_MLP,
        "n_layers": config.N_LAYERS,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "batch_size": config.BATCH_SIZE,
        "train_fraction": config.TRAIN_FRACTION,
        "steps": args.steps,
        "device": str(device),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    step = 0
    grok_step = None
    start = time.time()
    model.train()
    while step < args.steps:
        for tokens, targets in loader:
            if step >= args.steps:
                break
            tokens = tokens.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens), targets)
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0 or step == args.steps - 1:
                train_loss, train_acc = evaluate(
                    model, train_dataset, device
                )
                test_loss, test_acc = evaluate(model, test_dataset, device)
                print(
                    f"step={step} train_loss={train_loss:.6f} "
                    f"train_acc={train_acc:.6f} test_loss={test_loss:.6f} "
                    f"test_acc={test_acc:.6f} elapsed_s={time.time()-start:.1f}",
                    flush=True,
                )
                if grok_step is None and test_acc >= config.GROK_THRESHOLD:
                    grok_step = step
                if (
                    grok_step is not None
                    and step >= grok_step + config.EARLY_STOP_AFTER_GROK
                    and test_acc >= config.GROK_THRESHOLD
                ):
                    step = args.steps
                    break
            step += 1

    torch.save(model.state_dict(), run_dir / "model_final.pt")
    print(f"grok_step={grok_step}")
    print(f"output_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()

