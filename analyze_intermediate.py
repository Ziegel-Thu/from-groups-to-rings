"""Cross-fitted post-attention component-subspace intervention。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier

import config
from algebra import audit_projections, build_crt_projections
from model import ElementTransformer
from train import select_device


def build_groups(projections, target):
    """按全部非目标 CRT 坐标对环元素分组。"""
    groups = defaultdict(list)
    for element in range(len(projections[0])):
        key = tuple(
            int(projections[index][element])
            for index in range(len(projections))
            if index != target
        )
        groups[key].append(element)
    return [group for group in groups.values() if len(group) > 1]


def make_split(component_values, fraction, seed):
    """按目标 component 值分层留出完整 product elements。"""
    rng = np.random.RandomState(seed)
    elements = np.arange(len(component_values))
    held_out_elements = []
    for label in np.unique(component_values):
        group = elements[component_values == label].copy()
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * fraction)))
        n_test = min(n_test, len(group) - 1)
        held_out_elements.extend(group[:n_test].tolist())
    held_out = np.zeros(len(component_values), dtype=bool)
    held_out[held_out_elements] = True
    return held_out


def sample_interventions(
    table, projections, target, held_out, n_samples, seed
):
    """按 rebuttal 实验的固定种子抽样 held-out matched interventions。"""
    rng = np.random.RandomState(seed)
    groups = build_groups(projections, target)
    records = []
    attempts = 0
    target_values = projections[target]
    while len(records) < n_samples and attempts < n_samples * 1_000:
        attempts += 1
        group = groups[rng.randint(len(groups))]
        base, source = rng.choice(group, size=2, replace=False)
        right = rng.randint(len(table))
        base_output = int(table[base, right])
        source_output = int(table[source, right])
        if not held_out[base_output] or not held_out[source_output]:
            continue
        if target_values[base_output] == target_values[source_output]:
            continue
        records.append((base, source, right, base_output, source_output))
    if len(records) != n_samples:
        raise RuntimeError(
            f"component {target}: sampled {len(records)}/{n_samples} "
            "held-out interventions"
        )
    return np.asarray(records, dtype=np.int64), attempts


@torch.no_grad()
def collect_activations(model, ring_size, batch_size, device):
    """收集全部输入 pair 的 post-attention answer-token state。"""
    activations = []
    predictions = []
    pair_ids = np.arange(ring_size * ring_size)
    for start in range(0, len(pair_ids), batch_size):
        batch = pair_ids[start:start + batch_size]
        left = batch // ring_size
        right = batch % ring_size
        tokens = torch.as_tensor(
            np.stack(
                [left, np.full(len(batch), ring_size), right], axis=1
            ),
            dtype=torch.long,
            device=device,
        )
        logits, cache = model(tokens, return_cache=True)
        predictions.append(logits.argmax(-1).cpu().numpy())
        activations.append(cache["resid_mid"][:, 2, :].cpu().numpy())
    return np.concatenate(activations), np.concatenate(predictions)


def identify_probe_subspace(
    activations,
    products,
    component_values,
    held_out,
    variance_threshold,
    probe_kind,
):
    """在非 held-out product elements 上训练 probe 并提取权重子空间。"""
    train_mask = ~held_out[products]
    test_mask = held_out[products]
    labels = component_values[products]
    if probe_kind == "logistic":
        probe = LogisticRegression(
            max_iter=4_000, solver="lbfgs", C=1.0, random_state=42
        )
    elif probe_kind == "ridge":
        probe = RidgeClassifier(alpha=1.0)
    else:
        raise ValueError(f"unknown probe kind: {probe_kind}")
    probe.fit(activations[train_mask], labels[train_mask])
    weights = np.atleast_2d(probe.coef_.astype(np.float64))
    if len(weights) > 1:
        weights -= weights.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(
        weights, full_matrices=False
    )
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / energy.sum()
    dimension = int(np.searchsorted(cumulative, variance_threshold) + 1)
    basis = right_vectors[:dimension].T
    return basis, {
        "probe_kind": probe_kind,
        "retained_svd_energy": variance_threshold,
        "dimension": dimension,
        "probe_train_accuracy": float(
            (probe.predict(activations[train_mask]) == labels[train_mask]).mean()
        ),
        "probe_heldout_accuracy": float(
            (probe.predict(activations[test_mask]) == labels[test_mask]).mean()
        ),
        "probe_chance": float(
            np.max(np.bincount(labels[test_mask])) / test_mask.sum()
        ),
        "n_train_pairs": int(train_mask.sum()),
        "n_heldout_pairs": int(test_mask.sum()),
        "singular_values": singular_values.tolist(),
    }


def random_basis(width, dimension, rng):
    """生成同维 Haar-like 随机正交基。"""
    matrix = rng.normal(size=(width, dimension))
    basis, _ = np.linalg.qr(matrix)
    return basis[:, :dimension]


def summarize_random_metrics(metrics):
    """汇总全部与 rebuttal 表格对应的 random-subspace 指标。"""
    keys = (
        "target_accuracy",
        "full_counterfactual_accuracy",
        "non_target_preservation",
        "conditional_target_accuracy",
        "conditional_full_accuracy",
        "conditional_ntp",
    )
    summary = {}
    for key in keys:
        values = np.asarray(
            [item[key] for item in metrics if item[key] is not None],
            dtype=np.float64,
        )
        if len(values) == 0:
            summary[key] = {"mean": None, "std": None, "max": None}
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "max": float(values.max()),
        }
    return summary


@torch.no_grad()
def evaluate_basis(
    model,
    activations,
    predictions,
    records,
    projections,
    target,
    basis,
    batch_size,
    device,
):
    """在 held-out interventions 上评估一个固定 projector。"""
    ring_size = len(projections[0])
    projector = torch.as_tensor(
        basis @ basis.T, dtype=torch.float32, device=device
    )
    counts = defaultdict(int)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        base = batch[:, 0]
        source = batch[:, 1]
        right = batch[:, 2]
        base_outputs = batch[:, 3]
        cf_outputs = batch[:, 4]
        base_pair_ids = base * ring_size + right
        source_pair_ids = source * ring_size + right
        valid = (
            (predictions[base_pair_ids] == base_outputs)
            & (predictions[source_pair_ids] == cf_outputs)
        )

        base_states = torch.as_tensor(
            activations[base_pair_ids], dtype=torch.float32, device=device
        )
        source_states = torch.as_tensor(
            activations[source_pair_ids], dtype=torch.float32, device=device
        )
        patched_states = base_states + (source_states - base_states) @ projector
        tokens = torch.as_tensor(
            np.stack(
                [base, np.full(len(batch), ring_size), right], axis=1
            ),
            dtype=torch.long,
            device=device,
        )
        patched_predictions = model(
            tokens,
            patch_site="resid_mid",
            patch_state=patched_states,
            patch_position=2,
        ).argmax(-1).cpu().numpy()

        target_ok = (
            projections[target][patched_predictions]
            == projections[target][cf_outputs]
        )
        non_target_ok = np.ones(len(batch), dtype=bool)
        for component in range(len(projections)):
            if component != target:
                non_target_ok &= (
                    projections[component][patched_predictions]
                    == projections[component][cf_outputs]
                )
        exact = patched_predictions == cf_outputs
        counts["n"] += len(batch)
        counts["target"] += int(target_ok.sum())
        counts["full"] += int(exact.sum())
        counts["ntp"] += int(non_target_ok.sum())
        counts["conditional_n"] += int(valid.sum())
        counts["conditional_target"] += int((valid & target_ok).sum())
        counts["conditional_full"] += int((valid & exact).sum())
        counts["conditional_ntp"] += int((valid & non_target_ok).sum())

    def rate(key, denominator):
        return float(counts[key] / denominator) if denominator else None

    return {
        "n": int(counts["n"]),
        "target_accuracy": rate("target", counts["n"]),
        "full_counterfactual_accuracy": rate("full", counts["n"]),
        "non_target_preservation": rate("ntp", counts["n"]),
        "conditional_n": int(counts["conditional_n"]),
        "conditional_target_accuracy": rate(
            "conditional_target", counts["conditional_n"]
        ),
        "conditional_full_accuracy": rate(
            "conditional_full", counts["conditional_n"]
        ),
        "conditional_ntp": rate(
            "conditional_ntp", counts["conditional_n"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-interventions", type=int, default=2_000)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument(
        "--probe-kind", choices=("logistic", "ridge"), default="logistic"
    )
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.n_interventions < 1:
        parser.error("--n-interventions must be at least 1")
    if args.n_random < 1:
        parser.error("--n-random must be at least 1")
    if not 0.0 < args.variance_threshold <= 1.0:
        parser.error("--variance-threshold must be in (0, 1]")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be in (0, 1)")

    with open(args.run_dir / "config.json", encoding="utf-8") as handle:
        run_config = json.load(handle)
    algebra_name = run_config["algebra"]
    algebra_config = config.ALGEBRAS[algebra_name]
    table = np.load(args.run_dir / "multiplication_table.npy")
    projections = build_crt_projections(
        algebra_config["p"],
        algebra_config["group_order"],
        algebra_config["factors"],
    )
    print(f"projection_audit={audit_projections(table, projections)}")

    device = select_device(args.device)
    model = ElementTransformer(
        ring_size=run_config["ring_size"],
        d_model=run_config["d_model"],
        n_heads=run_config["n_heads"],
        d_mlp=run_config["d_mlp"],
        n_layers=run_config["n_layers"],
    ).to(device)
    state = torch.load(
        args.run_dir / "model_final.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    activations, predictions = collect_activations(
        model, len(table), args.batch_size, device
    )
    products = table.reshape(-1)
    print(f"full_table_accuracy={(predictions == products).mean():.6f}")

    results = {
        "algebra": algebra_name,
        "analysis_device": str(device),
        "analysis_seed": 42,
        "n_interventions_per_component": args.n_interventions,
        "n_random_subspaces": args.n_random,
        "split": "component-stratified held-out product elements",
        "probe_kind": args.probe_kind,
        "retained_svd_energy": args.variance_threshold,
        "projection_audit": audit_projections(table, projections),
        "components": {},
    }
    for target, name in enumerate(algebra_config["factor_names"]):
        held_out = make_split(
            projections[target], args.test_fraction, 42 + target
        )
        records, attempts = sample_interventions(
            table,
            projections,
            target,
            held_out,
            args.n_interventions,
            42 + target,
        )
        basis, identification = identify_probe_subspace(
            activations,
            products,
            projections[target],
            held_out,
            args.variance_threshold,
            args.probe_kind,
        )
        true_metrics = evaluate_basis(
            model,
            activations,
            predictions,
            records,
            projections,
            target,
            basis,
            args.batch_size,
            device,
        )
        rng = np.random.RandomState(1_000 + target)
        random_metrics = []
        for _ in range(args.n_random):
            random_metrics.append(evaluate_basis(
                model,
                activations,
                predictions,
                records,
                projections,
                target,
                random_basis(activations.shape[1], basis.shape[1], rng),
                args.batch_size,
                device,
            ))
        random_summary = summarize_random_metrics(random_metrics)
        results["components"][name] = {
            "identification": identification,
            "n_heldout_elements": int(held_out.sum()),
            "n_interventions": int(len(records)),
            "sampling_attempts": int(attempts),
            "true_subspace": true_metrics,
            "random_subspace": random_summary,
        }
        print(
            f"component={name} dim={identification['dimension']} "
            f"probe_heldout={identification['probe_heldout_accuracy']:.6f} "
            f"target={true_metrics['target_accuracy']:.6f} "
            f"full={true_metrics['full_counterfactual_accuracy']:.6f} "
            f"ntp={true_metrics['non_target_preservation']:.6f} "
            f"random_target_mean="
            f"{random_summary['target_accuracy']['mean']:.6f} "
            f"conditional_target={true_metrics['conditional_target_accuracy']} "
            f"conditional_full={true_metrics['conditional_full_accuracy']}",
            flush=True,
        )

    if args.probe_kind == "logistic" and args.variance_threshold == 0.95:
        output_name = "intermediate_subspace_metrics.json"
    else:
        threshold = int(round(100 * args.variance_threshold))
        output_name = f"intermediate_subspace_{args.probe_kind}_svd{threshold}.json"
    output_path = args.run_dir / output_name
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"output_path={output_path.resolve()}")


if __name__ == "__main__":
    main()
