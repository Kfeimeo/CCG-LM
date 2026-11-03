"""Single-device, exactly CE-token-budgeted training with atomic resumable state."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import warnings
import torch
from tqdm.auto import tqdm

from data.dataset import StatefulSampler, StoryDataset
from models.model import LatentForestLM, MODES, parameter_report
from utils import (atomic_json, atomic_torch_save, autocast_context, choose_device, device_info,
                   grad_norm, load_config, restore_rng, rng_state, set_seed, sync_file, to_device)


class TokenSchedule:
    """Warmup in optimizer steps, cosine endpoint fixed in *target tokens*."""
    def __init__(self, optimizer, warmup_steps, schedule_tokens, min_lr_ratio):
        self.optimizer = optimizer
        self.warmup_steps, self.schedule_tokens, self.min_lr_ratio = warmup_steps, schedule_tokens, min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.steps = self.tokens = 0
        self._apply()

    def _apply(self):
        if self.steps < self.warmup_steps:
            factor = (self.steps + 1) / max(1, self.warmup_steps)
        else:
            phase = min(1.0, self.tokens / self.schedule_tokens)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * phase)) / 2
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = peak * factor

    def step(self, tokens):
        self.steps += 1
        self.tokens += tokens
        self._apply()

    def state_dict(self):
        return {key: getattr(self, key) for key in ("steps", "tokens", "base_lrs", "warmup_steps", "schedule_tokens", "min_lr_ratio")}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._apply()


def make_optimizer(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        lr = cfg["decoder_lr"] if name.startswith("decoder.") else cfg["auxiliary_lr"]
        wd = cfg["weight_decay"] if p.ndim >= 2 else 0.0
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()],
                             betas=tuple(cfg["betas"]))


def make_schedule(optimizer, cfg):
    return TokenSchedule(optimizer, cfg["warmup_steps"], cfg["schedule_tokens"], cfg["min_lr_ratio"])


def train_update(model, sampler, optimizer, scheduler, tokens, micro_batch_size, device, precision, clip, progress=None, scaler=None):
    """Accumulate SUM losses / exact total tokens, never average micro-batch means."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    done, loss_sum, micro_batches = 0, 0.0, 0
    while done < tokens:
        batch, count = sampler.batch(micro_batch_size, tokens - done)
        batch = to_device(batch, device)
        with autocast_context(device, precision):
            result = model(**batch)
            loss = result["loss_sum"] / tokens
        (scaler.scale(loss) if scaler is not None and scaler.is_enabled() else loss).backward()
        loss_sum += float(result["loss_sum"].detach())
        done += count
        micro_batches += 1
        if progress is not None:
            progress.update(count)
    if scaler is not None and scaler.is_enabled(): scaler.unscale_(optimizer)
    syntax_grad = grad_norm(model.syntax.span_scorer if model.syntax is not None else None)
    composer_grad, reader_grad = grad_norm(model.composer), grad_norm(model.readers)
    enabled=scaler is not None and scaler.is_enabled()
    try:
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True))
    except RuntimeError:
        if not enabled: raise
        norm=float("inf")  # unscale_ has already recorded overflow for scaler.step.
    skipped=False
    if enabled:
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        skipped=scaler.get_scale()<before
        if not math.isfinite(norm) and not skipped: raise RuntimeError("Nonfinite norm without GradScaler overflow")
    else: optimizer.step()
    if not skipped: scheduler.step(done)
    metrics=dict(train_nll=loss_sum/done,step_tokens=done,micro_batches=micro_batches,
        span_scorer_grad_norm=syntax_grad,composer_grad_norm=composer_grad,reader_grad_norm=reader_grad,grad_norm=norm)
    if enabled:
        metrics.update(scaler_scale=scaler.get_scale(),skipped_step=skipped,nonfinite_gradient=not math.isfinite(norm))
        metrics={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in metrics.items()}
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, sampler, cfg, state, fingerprint, log_rows=None, scaler=None):
    from models.precision_policy import PrecisionPolicy
    expected=PrecisionPolicy(**cfg["precision"]).state_dict() if cfg.get("precision") else None
    actual=getattr(model,"precision_policy",None)
    if expected != (actual.state_dict() if actual is not None else None):
        raise ValueError("Checkpoint config and runtime precision policy differ")
    atomic_torch_save(dict(format_version=1, forest_implementation=model.forest.implementation if model.forest is not None else None,
                           forest_search_backend=getattr(model.forest, "search_backend", "eager") if model.forest is not None else None,
                           forest_reconstruction_mode=getattr(model.forest, "reconstruction_mode", None) if model.forest is not None else None,
                           model=model.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), rng=rng_state(),
                           precision_policy=getattr(model,"precision_policy",None).state_dict() if getattr(model,"precision_policy",None) else None,
                           grad_scaler=scaler.state_dict() if scaler is not None else None,
                           config=cfg, state=dict(state), data_fingerprint=fingerprint, log_rows=log_rows), path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, sampler=None, cfg=None, fingerprint=None, restore_random=True, scaler=None):
    # Only load trusted checkpoints: optimizer/Python RNG state requires pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if fingerprint is not None and checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("Data fingerprint differs from checkpoint")
    if cfg is not None:
        from models.precision_policy import PrecisionPolicy
        def normalized(value):return PrecisionPolicy(**value).state_dict() if value else None
        expected=normalized(cfg.get("precision"))
        if normalized(checkpoint["config"].get("precision"))!=expected:
            raise ValueError("Resume precision policy changed")
        actual=getattr(model,"precision_policy",None)
        if normalized(checkpoint.get("precision_policy"))!=expected or (actual.state_dict() if actual else None)!=expected:
            raise ValueError("Resume runtime precision policy differs")
        if checkpoint.get("grad_scaler") and scaler is None:
            raise ValueError("Resume FP16 checkpoint requires GradScaler")
        old_model = {key: value for key, value in checkpoint["config"]["model"].items() if key != "forest_implementation"}
        new_model = {key: value for key, value in cfg["model"].items() if key != "forest_implementation"}
        if old_model != new_model or checkpoint["config"]["seed"] != cfg["seed"]:
            raise ValueError("Resume model or seed changed")
        immutable = ["decoder_lr", "auxiliary_lr", "betas", "weight_decay", "warmup_steps",
                     "schedule_tokens", "min_lr_ratio", "effective_tokens", "grad_clip", "precision"]
        for key in immutable:
            if checkpoint["config"]["train"][key] != cfg["train"][key]:
                raise ValueError(f"Resume changes training trajectory: {key}")
        if checkpoint["config"]["train"]["micro_batch_size"] != cfg["train"]["micro_batch_size"]:
            warnings.warn("Micro-batch changed; data order is retained but dropout/numerical trajectory may differ")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("grad_scaler") is not None:
        scaler.load_state_dict(checkpoint["grad_scaler"])
    if sampler is not None:
        sampler.load_state_dict(checkpoint["sampler"])
    if restore_random:
        restore_rng(checkpoint["rng"])
    return checkpoint


class Logger:
    def __init__(self, directory, resume_step=None, resume_rows=None):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows = []
        file = self.path / "metrics.jsonl"
        if resume_rows is not None:
            self.rows = [row for row in resume_rows if row["step"] <= resume_step]
        elif file.exists() and resume_step is not None:
            self.rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            self.rows = [row for row in self.rows if row["step"] <= resume_step]
        elif file.exists():
            raise FileExistsError("Run already has logs; use --resume or a new --output-dir")
        self.flush()

    def append(self, row):
        self.rows.append(row)
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        csv_path = self.path / "metrics.csv"
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    def flush(self):
        # Reconcile any log tail ahead of the recovered atomic checkpoint.
        with (self.path / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        with (self.path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            if self.rows:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)


def run(cfg, resume=None):
    from evaluate import evaluate_loader
    from data.dataset import Collator
    from torch.utils.data import DataLoader
    tc = cfg["train"]
    device = choose_device(tc["device"])
    if device.type == "cpu":
        torch.set_num_threads(tc.get("cpu_threads", min(4, torch.get_num_threads())))
    set_seed(cfg["seed"], tc["deterministic"])
    info = device_info()
    print(json.dumps(info, indent=2))
    if device.type == "cuda" and tc["precision"] == "bf16" and not info["bf16_supported"]:
        warnings.warn("BF16 unsupported on this device; using FP32")
    train_data, dev_data = StoryDataset(cfg["data"]["cache_dir"]), StoryDataset(cfg["data"]["cache_dir"], "dev")
    manifest_cfg = train_data.manifest["spec"]["data"]
    for key in ("max_history", "max_words", "max_target_length"):
        if cfg["model"][key] < manifest_cfg[key]:
            raise ValueError(f"Model {key} is smaller than the cached data setting")
    if cfg["model"]["vocab_size"] < train_data.manifest["bpe_vocab_size"] or cfg["model"]["word_vocab_size"] < train_data.manifest["word_vocab_size"]:
        raise ValueError("Model vocabulary cannot represent cache token IDs")
    policy,scaler=None,None
    if cfg.get("precision"):
        from models.precision_policy import PrecisionPolicy,configure_model
        policy=PrecisionPolicy(**cfg["precision"]);policy.configure_backend()
    model = LatentForestLM(cfg["model"], cfg["seed"]).to(device)
    if policy is not None:
        configure_model(model,policy);scaler=policy.make_scaler(device)
    outer_precision=policy.outer_autocast if policy is not None else tc["precision"]
    optimizer, sampler = make_optimizer(model, tc), StatefulSampler(train_data, cfg["seed"])
    schedule = make_schedule(optimizer, tc)
    state = dict(step=0, target_tokens=0, train_seconds=0.0, elapsed_seconds=0.0, last_gradients={})
    resume_rows = None
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, schedule, sampler, cfg,
                                     train_data.manifest["fingerprint"], scaler=scaler)
        state, resume_rows = checkpoint["state"], checkpoint.get("log_rows")
        del checkpoint
    output = Path(tc["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    logger = Logger(output, state["step"] if resume else None, resume_rows)
    report = parameter_report(model)
    atomic_json(report, output / "parameters.json")
    atomic_json(cfg, output / "config.json")
    atomic_json(info, output / "environment.json")
    print(json.dumps(report, indent=2))
    loader = DataLoader(dev_data, batch_size=tc.get("eval_batch_size", 8), shuffle=False, collate_fn=Collator(),
                        num_workers=0, generator=torch.Generator().manual_seed(0))
    milestones = sorted(set(tc["checkpoint_tokens"]))
    base_elapsed, started = state["elapsed_seconds"], time.perf_counter()
    if tc["budget_tokens"] <= state["target_tokens"]:
        print("Checkpoint already meets requested budget")
        return state
    description = f"Train {model.mode}"
    progress = tqdm(total=tc["budget_tokens"], initial=state["target_tokens"], desc=description,
                    unit="tok", unit_scale=True, dynamic_ncols=True, mininterval=0.5, ascii=True,
                    disable=not tc.get("progress", True))
    try:
        while state["target_tokens"] < tc["budget_tokens"]:
            remaining = tc["budget_tokens"] - state["target_tokens"]
            next_marks = [mark - state["target_tokens"] for mark in milestones if mark > state["target_tokens"]]
            tokens = min([tc["effective_tokens"], remaining] + next_marks[:1])
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            lr_used = optimizer.param_groups[0]["lr"]
            metrics = train_update(model, sampler, optimizer, schedule, tokens, tc["micro_batch_size"],
                                   device, outer_precision, tc["grad_clip"], progress=progress, scaler=scaler)
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - tick
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
            if scaler is not None and scaler.is_enabled():
                state.setdefault("optimizer_updates",state["step"])
                state.setdefault("skipped_updates",0)
                state["optimizer_updates"]+=int(not metrics.get("skipped_step",False))
                state["skipped_updates"]+=int(metrics.get("skipped_step",False))
                metrics.update(optimizer_updates=state["optimizer_updates"],skipped_updates=state["skipped_updates"])
            state["step"] += 1
            state["target_tokens"] += tokens
            state["train_seconds"] += step_seconds
            state["last_gradients"] = {key: value for key, value in metrics.items() if "grad_norm" in key}
            final = state["target_tokens"] == tc["budget_tokens"]
            critical = state["target_tokens"] in milestones or final
            val_nll = None
            if critical or (tc["eval_every_steps"] and state["step"] % tc["eval_every_steps"] == 0):
                progress.set_description_str("Validate")
                val_nll = evaluate_loader(model, loader, device, outer_precision, max_batches=tc["eval_batches"], ablations=False)[0]["nll"]
                progress.set_description_str(description)
            state["elapsed_seconds"] = base_elapsed + time.perf_counter() - started
            row = dict(step=state["step"], target_tokens=state["target_tokens"], **metrics,
                       val_nll=val_nll, step_seconds=step_seconds, train_seconds=state["train_seconds"],
                       elapsed_seconds=state["elapsed_seconds"], gpu_hours=state["elapsed_seconds"] / 3600 if device.type == "cuda" else None,
                       target_tokens_per_second=tokens / step_seconds, peak_gpu_gib=peak,
                       decoder_lr=lr_used)
            logger.append(row)
            progress.set_postfix(step=state["step"], nll=f"{metrics['train_nll']:.4f}",
                                 val=f"{val_nll:.4f}" if val_nll is not None else "-",
                                 tok_s=f"{tokens / step_seconds:.0f}")
            if progress.disable:
                print(json.dumps(row, allow_nan=False), flush=True)
            else:
                progress.write(json.dumps(row, allow_nan=False))
            path = output / (f"tokens_{state['target_tokens']}.pt" if critical else "latest.pt")
            if critical or state["step"] % tc.get("save_every_steps", 100) == 0:
                save_checkpoint(path, model, optimizer, schedule, sampler, cfg, state, train_data.manifest["fingerprint"], logger.rows, scaler=scaler)
            if critical:
                sync_file(path, tc.get("drive_dir"))
                for file in ("metrics.csv", "metrics.jsonl", "config.json", "parameters.json", "environment.json"):
                    sync_file(output / file, tc.get("drive_dir"))
    except torch.OutOfMemoryError as error:
        suggestion = max(1, tc["micro_batch_size"] // 2)
        raise RuntimeError(f"CUDA OOM. Resume the latest complete checkpoint with --micro-batch-size {suggestion}; "
                           "keep effective_tokens fixed. If already 1, reduce max_words or model size in a NEW run/cache.") from error
    finally:
        progress.close()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir")
    parser.add_argument("--no-progress", action="store_true", help="Disable the target-token tqdm progress bar")
    parser.add_argument("--forest-implementation", choices=["reference", "score_first", "lazy_semantic", "lazy_tensorized_a", "lazy_tensorized", "lazy_tensorized_score_reconstruct"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, attr in [("budget_tokens", "budget_tokens"), ("micro_batch_size", "micro_batch_size"), ("output_dir", "output_dir"), ("device", "device")]:
        if getattr(args, attr) is not None:
            cfg["train"][key] = getattr(args, attr)
    if args.mode:
        cfg["model"]["mode"] = args.mode
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    if args.no_progress:
        cfg["train"]["progress"] = False
    if args.forest_implementation is not None:
        cfg["model"]["forest_implementation"] = args.forest_implementation
    if cfg["train"]["budget_tokens"] <= 0 or cfg["train"]["effective_tokens"] <= 0 or cfg["train"]["micro_batch_size"] <= 0:
        parser.error("Token budgets and micro-batch size must be positive")
    run(cfg, args.resume)


if __name__ == "__main__":
    main()
