"""Plan, train, or time local-CNN controls using the existing LM training path."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import pandas as pd
import torch

from models.model import LatentForestLM, parameter_report
from models.precision_policy import PrecisionPolicy, configure_model
from utils import atomic_json, autocast_context, device_info, load_config, set_seed, to_device


VARIANTS = {
    "no_memory": ("no_memory", None),
    "pooling4": ("pooling4", None),
    "cnn_k1": ("seq_cnn4", 1),
    "cnn_k3": ("seq_cnn4", 3),
    "cnn_k5": ("seq_cnn4", 5),
    "fixed_tree4": ("fixed_tree4", None),
    "forest_scalar4": ("forest_scalar4", None),
    "forest_vector4": ("forest_vector4", None),
}
DEFAULT_VARIANTS = ["no_memory", "pooling4", "cnn_k1", "cnn_k3", "forest_scalar4"]


def variant_config(base, variant, seed, root):
    cfg = copy.deepcopy(base)
    mode, kernel = VARIANTS[variant]
    cfg["name"] = f"{base['name']}_{variant}"
    cfg["seed"] = seed
    cfg["model"]["mode"] = mode
    if kernel is not None:
        cfg["model"]["cnn_kernel_size"] = kernel
    cfg["train"]["output_dir"] = str(root / f"{variant}_seed{seed}")
    return cfg


def benchmark(cfg, batch, args):
    from train import make_optimizer
    set_seed(cfg["seed"])
    policy = PrecisionPolicy(**cfg["precision"]) if cfg.get("precision") else None
    if policy:
        policy.configure_backend()
    model = LatentForestLM(cfg["model"], cfg["seed"]).cuda().train()
    if policy:
        configure_model(model, policy)
    precision = policy.outer_autocast if policy else cfg["train"]["precision"]
    scaler = policy.make_scaler("cuda") if policy else None
    optimizer = make_optimizer(model, cfg["train"])

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch.device("cuda"), precision):
            loss = model(**batch)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite benchmark loss")
        (scaler.scale(loss) if scaler is not None and scaler.is_enabled() else loss).backward()
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        tick = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - tick))
    report = dict(samples_ms=samples, median_step_ms=statistics.median(samples),
                  target_tokens=int(batch["labels"].ne(-100).sum()),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  precision_policy=policy.state_dict() if policy else None,
                  parameters=parameter_report(model)["total"])
    report["target_tokens_per_second"] = report["target_tokens"] * 1000 / report["median_step_ms"]
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablations/seq_cnn.yaml")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--effective-tokens", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--cnn-layers", type=int)
    parser.add_argument("--cnn-hidden-size", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/local_cnn_ablation")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true", help="Run the planned training jobs sequentially")
    action.add_argument("--benchmark", action="store_true", help="CUDA resident-batch full-update timings only")
    action.add_argument("--dry-run", action="store_true", help="Only save the plan and parameter table (default)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate final checkpoints on dev after training")
    parser.add_argument("--source", choices=["synthetic", "cache"], default="synthetic")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--words", type=int, default=24)
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.evaluate and not args.train:
        parser.error("--evaluate requires --train")
    for key in ("budget_tokens", "effective_tokens", "micro_batch_size", "cnn_layers", "cnn_hidden_size",
                "batch_size", "histories", "words", "target_length", "warmup", "repeats"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Variants and seeds must not contain duplicates")
    base = load_config(args.config)
    for key in ("budget_tokens", "effective_tokens", "micro_batch_size"):
        if getattr(args, key) is not None:
            base["train"][key] = getattr(args, key)
    for key in ("cnn_layers", "cnn_hidden_size"):
        if getattr(args, key) is not None:
            base["model"][key] = getattr(args, key)
    base["train"]["device"] = args.device
    if args.cache_dir:
        base["data"]["cache_dir"] = args.cache_dir
    root = Path(args.output_dir).resolve()
    plan, parameters = [], []
    for variant in args.variants:
        cfg = variant_config(base, variant, args.seeds[0], root)
        with torch.device("meta"):
            model = LatentForestLM(cfg["model"], cfg["seed"])
        report = parameter_report(model)
        parameters.append(dict(variant=variant, mode=model.mode, parameters=report["total"],
                               receptive_field=model.seq_cnn.receptive_field if model.seq_cnn else None,
                               **report["modules"]))
        del model
        for seed in args.seeds:
            cfg = variant_config(base, variant, seed, root)
            directory = Path(cfg["train"]["output_dir"])
            if args.train and ((directory / "metrics.jsonl").exists() or any(directory.glob("*.pt"))):
                parser.error(f"Existing run: {directory}; use a new --output-dir or resume via train.py")
            plan.append(dict(variant=variant, seed=seed, config=cfg))
    atomic_json(dict(arguments=vars(args), runs=plan,
                     note="Same CE-token budget, data, schedule and decoder architecture; auxiliary parameter counts differ. "
                          "Formal NLL comparisons require adequate training and repeated seeds."), root / "plan.json")
    pd.DataFrame(parameters).fillna(0).to_csv(root / "parameters.csv", index=False)
    print(pd.DataFrame(parameters).fillna(0).to_string(index=False), flush=True)
    if args.benchmark:
        if args.device != "cuda" or not torch.cuda.is_available():
            parser.error("--benchmark requires --device cuda and CUDA PyTorch")
        from profile_performance import make_batch
        torch.set_num_threads(4)
        if args.histories > base["model"]["max_history"] or args.words > base["model"]["max_words"] or args.target_length > base["model"]["max_target_length"]:
            parser.error("Benchmark shape exceeds model capacity")
        # One identical resident input batch for every architecture and seed.
        set_seed(args.seeds[0])
        batch = to_device(make_batch(base, args.batch_size, args.histories, args.words,
                                    args.target_length, args.source), torch.device("cuda"))
        rows = []
        atomic_json(dict(environment=device_info(), arguments=vars(args),
                         history_shape=list(batch["history_ids"].shape), input_shape=list(batch["input_ids"].shape),
                         note="Independent fresh architectures, sequential timing, no training-quality claim. "
                              "Includes forward/backward/clip/AdamW; excludes input loading, evaluation and checkpoint IO."),
                    root / "benchmark_environment.json")
        for entry in plan:
            report = benchmark(entry["config"], batch, args)
            atomic_json(report, root / f"{entry['variant']}_seed{entry['seed']}_samples.json")
            row = dict(variant=entry["variant"], seed=entry["seed"],
                       **{k: v for k, v in report.items() if k not in ("samples_ms", "precision_policy")})
            rows.append(row)
            pd.DataFrame(rows).to_csv(root / "benchmark.csv", index=False)
            print(json.dumps(row), flush=True)
    elif args.train:
        for entry in plan:
            cfg = entry["config"]
            config_path = root / f"{entry['variant']}_seed{entry['seed']}.json"
            atomic_json(cfg, config_path)
            subprocess.run([sys.executable, "train.py", "--config", str(config_path)], check=True)
            if args.evaluate:
                directory = Path(cfg["train"]["output_dir"])
                checkpoint = directory / f"tokens_{cfg['train']['budget_tokens']}.pt"
                subprocess.run([sys.executable, "evaluate.py", "--checkpoints", str(checkpoint),
                                "--cache-dir", cfg["data"]["cache_dir"], "--output-dir", str(directory / "dev"),
                                "--split", "dev", "--device", args.device], check=True)
    else:
        print(f"Plan saved to {root / 'plan.json'}. Add --train or --benchmark to execute.")


if __name__ == "__main__":
    main()
