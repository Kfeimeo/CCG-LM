"""NLL, memory interventions, structural diagnostics, paired bootstrap and curves."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.dataset import Collator, StoryDataset
from models.model import LatentForestLM, parameter_report
from utils import atomic_json, autocast_context, choose_device, device_info, to_device


def tree_edges(tree):
    """Count internal child edges by side; a balanced tree contributes both."""
    if isinstance(tree, int):
        return 0, 0
    ll, lr = tree_edges(tree[0])
    rl, rr = tree_edges(tree[1])
    return ll + rl + int(isinstance(tree[0], tuple)), lr + rr + int(isinstance(tree[1], tuple))


def memory_diagnostics(memory, attention, target_mask, sums):
    if memory is None:
        return
    mask = memory["candidate_mask"] & memory["history_mask"].unsqueeze(-1)
    active = mask.any(-1)
    count = int(active.sum())
    sums["sentences"] += count
    sums["root_candidates"] += int(mask.sum())
    prior = memory["log_prior"].float().exp() * mask
    entropy = -(prior * memory["log_prior"].float()).sum(-1)
    sums["root_entropy"] += float(entropy[active].sum())
    sem = F.normalize(memory["semantic"].float(), dim=-1)
    cosine = sem @ sem.transpose(-2, -1)
    k = mask.shape[-1]
    upper = torch.ones(k, k, device=mask.device, dtype=torch.bool).triu(1)
    pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper
    sums["cosine_sum"] += float(cosine[pairs].sum())
    sums["cosine_pairs"] += int(pairs.sum())
    # Fraction of candidates with at least one >=0.99 cosine duplicate, among
    # sentences with >=2 candidates. Singletons are excluded from the denominator.
    both = mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~torch.eye(k, device=mask.device, dtype=torch.bool)
    eligible = mask & (mask.sum(-1, keepdim=True) >= 2)
    duplicate = ((cosine >= 0.99) & both).any(-1) & eligible
    sums["collapsed"] += int(duplicate.sum())
    sums["collapse_eligible"] += int(eligible.sum())
    diag = memory.get("diagnostics", {})
    if "span_count" in diag:
        sums["span_candidates"] += float(diag["span_candidates_sum"])
        sums["spans"] += float(diag["span_count"])
    for layer in attention:
        # Entropy over all valid history/candidate slots, averaged over heads
        # and true target positions (and separately over reader layers).
        entropy = -(layer.float() * layer.float().clamp_min(1e-30).log()).sum(-1)
        valid = target_mask[:, None, :].expand_as(entropy) & active.any(-1)[:, None, None]
        sums["attention_entropy"] += float(entropy[valid].sum())
        sums["attention_queries"] += int(valid.sum())
    for candidates in memory.get("trees", []):
        for tree in candidates:
            left, right = tree_edges(tree)
            sums["left_edges"] += left
            sums["right_edges"] += right


@torch.no_grad()
def evaluate_loader(model, loader, device, precision="bf16", max_batches=None, ablations=True,
                    diagnostics=False, top1=False):
    previous = model.training
    model.eval()
    variants = ["normal", "memory_off", "shuffled_memory"] if ablations and model.mode != "no_memory" else ["normal"]
    if top1 and model.forest is not None:
        variants.append("forest_top1_eval")
    sums, structure, stories = defaultdict(float), defaultdict(float), defaultdict(lambda: defaultdict(float))
    normal_seconds, batches, shuffled_tokens = 0.0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = itertools.islice(loader, max_batches) if max_batches else iter(loader)
    try:
        for batch in iterator:
            batch = to_device(batch, device)
            valid = batch["labels"].ne(-100)
            tokens = int(valid.sum())
            # Last singleton is still included in normal/off NLL. Shuffled NLL
            # differences use only the explicitly reported matched subset.
            for variant in variants:
                if variant == "shuffled_memory" and len(batch["input_ids"]) == 1:
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                with autocast_context(device, precision):
                    result = model(**batch, ablation=variant, return_attention=diagnostics and variant == "normal",
                                   return_trees=diagnostics and variant == "normal")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                if variant != "shuffled_memory":
                    sums[variant + "_loss"] += float(result["loss_sum"])
                    sums[variant + "_tokens"] += tokens
                if variant == "normal":
                    normal_seconds += duration
                    losses = result["token_losses"]
                    early = torch.arange(valid.shape[1], device=device)[None, :] < 8
                    sums["early_loss"] += float(losses[valid & early].sum())
                    sums["early_tokens"] += int((valid & early).sum())
                    sums["late_loss"] += float(losses[valid & ~early].sum())
                    sums["late_tokens"] += int((valid & ~early).sum())
                    for i, story in enumerate(batch["story_ids"]):
                        stories[story]["loss_sum"] += float(losses[i].sum())
                        stories[story]["tokens"] += int(valid[i].sum())
                    if diagnostics:
                        memory_diagnostics(result["memory"], result["attention"], valid, structure)
                    normal_losses = losses
                if variant == "shuffled_memory":
                    matched = valid & result["memory"]["shuffle_valid"][:, None]
                    shuffled_tokens += int(matched.sum())
                    sums["shuffled_memory_loss"] += float(result["token_losses"][matched].sum())
                    sums["shuffled_matched_normal_loss"] += float(normal_losses[matched].sum())
            batches += 1
    finally:
        model.train(previous)
    if sums["normal_tokens"] == 0:
        raise ValueError("Evaluation contains no target tokens")
    nll = sums["normal_loss"] / sums["normal_tokens"]
    metrics = dict(nll=nll, perplexity=math.exp(nll) if nll < 700 else None,
                   first8_nll=sums["early_loss"] / sums["early_tokens"] if sums["early_tokens"] else None,
                   later_nll=sums["late_loss"] / sums["late_tokens"] if sums["late_tokens"] else None,
                   target_tokens=int(sums["normal_tokens"]), batches=batches,
                   normal_forward_seconds=normal_seconds,
                   forward_target_tokens_per_second=sums["normal_tokens"] / normal_seconds,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None)
    if sums["memory_off_tokens"]:
        off = sums["memory_off_loss"] / sums["memory_off_tokens"]
        metrics.update(memory_off_nll=off, memory_off_minus_normal_nll=off - nll)
    if shuffled_tokens:
        shuffled = sums["shuffled_memory_loss"] / shuffled_tokens
        metrics.update(shuffled_memory_nll=shuffled, shuffled_target_tokens=shuffled_tokens,
                       shuffled_minus_normal_nll=(sums["shuffled_memory_loss"] - sums["shuffled_matched_normal_loss"]) / shuffled_tokens)
    if sums["forest_top1_eval_tokens"]:
        metrics["forest_top1_eval_nll"] = sums["forest_top1_eval_loss"] / sums["forest_top1_eval_tokens"]
    if diagnostics:
        def ratio(a, b):
            return structure[a] / structure[b] if structure[b] else None
        branches = structure["left_edges"] + structure["right_edges"]
        metrics.update(mean_span_candidates=ratio("span_candidates", "spans"),
                       mean_root_candidates=ratio("root_candidates", "sentences"),
                       root_candidate_entropy=ratio("root_entropy", "sentences"),
                       mean_candidate_cosine_similarity=ratio("cosine_sum", "cosine_pairs"),
                       candidate_collapse_rate=ratio("collapsed", "collapse_eligible"),
                       candidate_attention_entropy=ratio("attention_entropy", "attention_queries"),
                       left_branch_ratio=structure["left_edges"] / branches if branches else None,
                       right_branch_ratio=structure["right_edges"] / branches if branches else None)
    rows = [dict(story_id=story, **stats) for story, stats in sorted(stories.items())]
    return metrics, rows


def paired_bootstrap(a, b, iterations=2000, seed=2026, confidence=0.95):
    """Token-weighted NLL difference, paired resampling of whole stories."""
    a, b = a.set_index("story_id").sort_index(), b.set_index("story_id").sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError("Require exactly one row per identical paired story")
    if not np.array_equal(a.tokens.to_numpy(), b.tokens.to_numpy()):
        raise ValueError("Paired stories must have identical target token counts")
    if "data_fingerprint" in a and "data_fingerprint" in b:
        if set(a.data_fingerprint) != set(b.data_fingerprint):
            raise ValueError("Bootstrap data caches differ")
    if len(a) < 2 or iterations < 2:
        raise ValueError("Bootstrap requires >=2 stories and iterations")
    diff = a.loss_sum.to_numpy() - b.loss_sum.to_numpy()
    tokens = a.tokens.to_numpy()
    rng = np.random.default_rng(seed)
    values = np.empty(iterations)
    for i in range(iterations):
        selection = rng.integers(0, len(a), size=len(a))
        values[i] = diff[selection].sum() / tokens[selection].sum()
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(values, [tail, 1 - tail])
    return dict(difference_a_minus_b=float(diff.sum() / tokens.sum()), ci_low=float(lo), ci_high=float(hi),
                confidence=confidence, stories=len(a), iterations=iterations, seed=seed)


def learning_efficiency(frame, nll_threshold=None, lower=5_000_000, upper=100_000_000):
    clean = frame.dropna(subset=["val_nll"]).sort_values("target_tokens").drop_duplicates("target_tokens", keep="last")
    result = dict(aulc_5m_100m=None, normalized_aulc_5m_100m=None, aulc_coverage="insufficient checkpoints",
                  threshold_nll=nll_threshold, threshold_target_tokens=None, threshold_seconds=None)
    if len(clean) == 0:
        return result
    x, y = clean.target_tokens.to_numpy(), clean.val_nll.to_numpy()
    if x[0] <= lower and x[-1] >= upper:
        inside = (x > lower) & (x < upper)
        xs = np.concatenate(([lower], x[inside], [upper]))
        ys = np.interp(xs, x, y)
        area = float(np.trapezoid(ys, xs))
        result.update(aulc_5m_100m=area, normalized_aulc_5m_100m=area / (upper - lower), aulc_coverage="complete; linear interpolation between observations")
    if nll_threshold is not None:
        crossings = np.flatnonzero(y <= nll_threshold)
        if len(crossings):
            idx = int(crossings[0])
            # First observed crossing; do not invent an unobserved training time.
            result["threshold_target_tokens"] = int(x[idx])
            result["threshold_seconds"] = float(clean.iloc[idx].elapsed_seconds)
    return result


def plot_logs(paths, output_dir, threshold=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        label = str(Path(path).parent)
        valid = frame.dropna(subset=["val_nll"])
        axes[0].plot(valid.target_tokens, valid.val_nll, marker="o", label=label)
        if "gpu_hours" in valid and valid.gpu_hours.notna().any():
            axes[1].plot(valid.gpu_hours, valid.val_nll, marker="o", label=label)
        rows.append(dict(run=label, **learning_efficiency(frame, threshold),
                         final_target_tokens=int(frame.target_tokens.iloc[-1]),
                         train_seconds=float(frame.train_seconds.iloc[-1]),
                         elapsed_seconds=float(frame.elapsed_seconds.iloc[-1]),
                         mean_target_tokens_per_second=float(frame.step_tokens.sum() / frame.step_seconds.sum()),
                         peak_gpu_gib=float(frame.peak_gpu_gib.max()) if frame.peak_gpu_gib.notna().any() else None))
    axes[0].set_xlabel("CE target tokens")
    axes[1].set_xlabel("GPU hours (active wall time)")
    for ax in axes:
        ax.set_ylabel("Development NLL (nats/token)")
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "learning_efficiency.csv", index=False)
    return rows


def evaluate_checkpoints(paths, cache_dir, output_dir, split="dev", batch_size=8, device_name="auto",
                         max_batches=None, diagnostics=False, top1=False):
    from train import load_checkpoint
    device = choose_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data = StoryDataset(cache_dir, split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        source_fingerprint=cfg.get("study",{}).get("source_cache_fingerprint",checkpoint["data_fingerprint"])
        if source_fingerprint != data.manifest["fingerprint"]:
            raise ValueError("Evaluation cache differs from checkpoint")
        model = LatentForestLM(cfg["model"], cfg["seed"]).to(device)
        outer_precision=cfg["train"]["precision"]
        if cfg.get("precision"):
            from models.precision_policy import PrecisionPolicy,configure_model
            policy=PrecisionPolicy(**cfg["precision"]);policy.configure_backend();configure_model(model,policy)
            outer_precision=policy.outer_autocast
        model.load_state_dict(checkpoint["model"])
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=Collator(), num_workers=0)
        metrics, stories = evaluate_loader(model, loader, device, outer_precision, max_batches,
                                           diagnostics=diagnostics, top1=top1)
        identifier = Path(path).parent.name + "_" + Path(path).stem
        gradients = checkpoint["state"].get("last_gradients", {})
        row = dict(checkpoint=str(path), mode=model.mode, seed=cfg["seed"], split=split,
                   data_fingerprint=checkpoint["data_fingerprint"],
                   trained_target_tokens=checkpoint["state"]["target_tokens"],
                   parameters=parameter_report(model)["total"], **gradients, **metrics)
        all_rows.append(row)
        for story in stories:
            story.update(data_fingerprint=checkpoint["data_fingerprint"], checkpoint=str(path))
        pd.DataFrame(stories).to_csv(output / f"{identifier}_stories.csv", index=False)
        atomic_json(row, output / f"{identifier}.json")
        print(json.dumps(row, indent=2, allow_nan=False))
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(all_rows).to_csv(output / "summary.csv", index=False)
    atomic_json(device_info(), output / "environment.json")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--cache-dir", default="cache/demo")
    parser.add_argument("--output-dir", default="runs/evaluation")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--top1", action="store_true")
    parser.add_argument("--logs", nargs="+")
    parser.add_argument("--nll-threshold", type=float)
    parser.add_argument("--bootstrap", nargs=2, metavar=("A_STORIES_CSV", "B_STORIES_CSV"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    if not any([args.checkpoints, args.logs, args.bootstrap]):
        parser.error("Supply --checkpoints, --logs or --bootstrap")
    if args.checkpoints:
        evaluate_checkpoints(args.checkpoints, args.cache_dir, args.output_dir, args.split, args.batch_size,
                             args.device, args.max_batches, args.diagnostics, args.top1)
    if args.logs:
        plot_logs(args.logs, args.output_dir, args.nll_threshold)
    if args.bootstrap:
        result = paired_bootstrap(*(pd.read_csv(path) for path in args.bootstrap), iterations=args.bootstrap_iterations)
        atomic_json(result, Path(args.output_dir) / "paired_bootstrap.json")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
