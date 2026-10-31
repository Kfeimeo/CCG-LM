"""Paired V2/V2.5 benchmark with identical weight/AdamW/RNG snapshots.

Uses profile_performance.py's regular and staged steps. All snapshot restores,
audits and profiler runs occur outside uninstrumented timing. Arms alternate
order to reduce clock/temperature bias. No architecture/search budget changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch
import pandas as pd

from models.model import LatentForestLM
from profile_performance import make_batch, regular_step, staged_step
from train import make_optimizer
from utils import atomic_json, autocast_context, device_info, load_config, restore_rng, rng_state, set_seed, sha256, to_device

CASES = [
    ("smoke_b2_w8", "smoke", 2, 8, False),
    ("smoke_b2_w16", "smoke", 2, 16, False),
    ("smoke_b2_w24", "smoke", 2, 24, False),
    ("smoke_b8_w24", "smoke", 8, 24, False),
    ("main_b1_w24", "main_125m", 1, 24, False),
    ("main_b4_w24", "main_125m", 4, 24, False),
    ("real_b2", "smoke", 2, 24, True),
]


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def snapshot_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore(model, optimizer, state, implementation):
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    model.forest.implementation = implementation
    torch.cuda.synchronize()


def operator_counts(model, batch, optimizer, precision, capture_kernel_names=False):
    contexts = []
    def before(module, args):
        ctx = torch.profiler.record_function("CKY_FORWARD")
        ctx.__enter__()
        contexts.append(ctx)
    def after(module, args, result):
        contexts.pop().__exit__(None, None, None)
    handles = [model.forest.register_forward_pre_hook(before), model.forest.register_forward_hook(after)]
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            regular_step(model, batch, optimizer, precision)
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    forward, full = Counter(), Counter()
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        name = event.name
        if name not in ("aten::stack", "aten::cat") and not name.startswith("cudaLaunchKernel"):
            continue
        key = "cudaLaunchKernel" if name.startswith("cudaLaunchKernel") else name
        full[key] += 1
        parent = event.cpu_parent
        while parent is not None:
            if parent.name == "CKY_FORWARD":
                forward[key] += 1
                break
            parent = parent.cpu_parent
    metrics = {f"{scope}_{name.replace('aten::', '')}": counter[name]
            for scope, counter in (("cky_forward", forward), ("full_step", full))
            for name in ("aten::stack", "aten::cat", "cudaLaunchKernel")}
    intervals=sorted((e.time_range.start,e.time_range.end) for e in profiler.events()
                     if e.device_type == torch.autograd.DeviceType.CUDA)
    gaps=[]
    if intervals:
        end=intervals[0][1]
        for a,b in intervals[1:]:
            if a>end: gaps.append(a-end)
            end=max(end,b)
    metrics.update(device_idle_gaps_over_50us=sum(x>50 for x in gaps),
                   device_idle_gap_total_ms=sum(gaps)/1000,device_idle_gap_max_ms=max(gaps,default=0)/1000)
    if capture_kernel_names:
        metrics["cuda_kernel_names"]=sorted({e.name for e in profiler.events() if e.device_type==torch.autograd.DeviceType.CUDA and (any(t in e.name.lower() for t in ("gemm","matmul")) or ("mma" in e.name.lower() and "gamma" not in e.name.lower()))})
    return metrics


def autograd_nodes(tensors):
    seen=set(); stack=[x.grad_fn for x in tensors if x.grad_fn is not None]
    while stack:
        fn=stack.pop()
        if fn in seen: continue
        seen.add(fn);stack.extend(x for x,_ in fn.next_functions if x is not None)
    return len(seen)


def run_case(case, args):
    name, config, batch_size, words, real = case
    set_seed(42)
    cfg = load_config(f"configs/{config}.yaml")
    cfg["model"].update(mode="forest_vector4", forest_implementation="score_first")
    cfg["data"]["cache_dir"] = args.cache_dir
    model = LatentForestLM(cfg["model"]).cuda().train()
    optimizer = make_optimizer(model, cfg["train"])
    batch = to_device(make_batch(cfg, batch_size, 4, words, 128, "cache" if real else "synthetic"), torch.device("cuda"))
    for _ in range(args.warmup):
        regular_step(model, batch, optimizer, args.precision)
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    state = dict(model=cpu_snapshot(model.state_dict()), optimizer=cpu_snapshot(optimizer.state_dict()), rng=rng_state())
    weight_hash = snapshot_hash(state["model"])
    # Also warm V2's lookup cache and its forward/backward kernels, then restore
    # the very same shared checkpoint snapshot before EVERY measured sample.
    restore(model, optimizer, state, "lazy_semantic")
    for _ in range(args.warmup):
        regular_step(model, batch, optimizer, args.precision)
    torch.cuda.synchronize()
    samples = {key: [] for key in ("score_first", "lazy_semantic")}
    peaks = {key: [] for key in samples}
    for repeat in range(args.repeats):
        order = ("score_first", "lazy_semantic") if repeat % 2 == 0 else ("lazy_semantic", "score_first")
        for implementation in order:
            restore(model, optimizer, state, implementation)
            torch.cuda.reset_peak_memory_stats()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter()
            a.record()
            regular_step(model, batch, optimizer, args.precision)
            b.record()
            torch.cuda.synchronize()
            samples[implementation].append(dict(wall_ms=(time.perf_counter() - tick) * 1000,
                                                cuda_envelope_ms=a.elapsed_time(b)))
            peaks[implementation].append(torch.cuda.max_memory_allocated() / 2**30)
    rows, ops, diagnostics = [], [], {}
    workloads = {}
    tokens = int(batch["labels"].ne(-100).sum())
    for implementation in samples:
        parts = []
        for _ in range(3):
            restore(model, optimizer, state, implementation)
            parts.append(staged_step(model, batch, optimizer, args.precision))
        diagnostics[implementation] = parts
        restore(model, optimizer, state, implementation)
        # Materialization accounting is allocation-element accounting for
        # representation staging, not peak bytes, views or Composer internals.
        model.forest.collect_audit = True
        shapes = []
        handle = model.composer.register_forward_pre_hook(lambda module, inputs: shapes.append(list(inputs[0].shape)))
        try:
            with autocast_context(torch.device("cuda"), args.precision):
                output = model(**batch)
            audit = dict(output["memory"]["materialization_audit"])
            graph_counts=dict(forest_autograd_nodes=autograd_nodes([output["memory"][k] for k in ("semantic","vector_coeff","log_prior")]),
                              score_autograd_nodes=autograd_nodes([output["memory"]["log_prior"]]),
                              full_autograd_nodes=autograd_nodes([output["loss"]]))
        finally:
            model.forest.collect_audit = False
            handle.remove()
        calls, candidate_rows = len(shapes),sum(math.prod(s[:-1]) for s in shapes)
        assert audit.get("composer_calls",audit.get("v25_composer_calls")) == calls
        assert audit.get("composer_input_candidate_rows",audit.get("v25_actual_composer_rows")) == candidate_rows
        workloads[implementation] = dict(audit=audit,shapes=shapes,**graph_counts)
        phase_samples=[]
        if implementation == "lazy_semantic":
            model.forest.collect_timing=True
            try:
                for _ in range(args.repeats):
                    restore(model,optimizer,state,implementation)
                    with torch.no_grad(),autocast_context(torch.device("cuda"),args.precision):
                        probe=model(**batch)
                    phase_samples.append(model.forest.last_workload.copy());del probe
            finally: model.forest.collect_timing=False
            workloads[implementation]["phase_samples"]=phase_samples
        del output
        restore(model, optimizer, state, implementation)
        counts = operator_counts(model, batch, optimizer, args.precision)
        ops.append(dict(case=name, implementation=implementation, **counts, **graph_counts, composer_calls=calls, composer_rows=candidate_rows))
        wall = [row["wall_ms"] for row in samples[implementation]]
        fwd = [part["module/cky"]["cuda_envelope_ms"] for part in parts]
        backward = [part["phase/cky_backward"]["wall_ms"] for part in parts]
        rows.append(dict(case=name, config=config, implementation=implementation, precision=args.precision,
                         batch=batch_size, history_words_requested=words, history_words_actual=batch["history_ids"].shape[-1],
                         histories=4, target_length=batch["labels"].shape[-1], ce_target_tokens=tokens,
                         k=cfg["model"]["candidates"], r=cfg["model"]["max_compositions"],
                         weight_snapshot_sha256=weight_hash, step_median_ms=statistics.median(wall),
                         step_min_ms=min(wall), step_max_ms=max(wall), target_tokens_per_second=tokens * 1000 / statistics.median(wall),
                         peak_allocated_gib=max(peaks[implementation]),
                         cky_forward_median_ms=statistics.median(fwd), cky_forward_min_ms=min(fwd), cky_forward_max_ms=max(fwd),
                         cky_backward_median_ms=statistics.median(backward), cky_backward_min_ms=min(backward), cky_backward_max_ms=max(backward)))
        if phase_samples:
            for metric in ("score_search_cuda_ms","semantic_cuda_ms","topology_d2h_blocking_ms","dag_cpu_ms","index_h2d_host_ms","index_h2d_cuda_ms"):
                values=[x[metric] for x in phase_samples]
                rows[-1].update({metric+"_median":statistics.median(values),metric+"_min":min(values),metric+"_max":max(values)})
        rows[-1].update(composer_rows=candidate_rows,composer_calls=calls,**graph_counts)
        print(json.dumps(rows[-1]), flush=True)
    assert ops[1]["composer_rows"] <= ops[0]["composer_rows"]
    assert ops[1]["composer_calls"] <= ops[0]["composer_calls"]
    root = Path(args.output_dir) / "raw" / name
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(workloads, root / "composer_workload.json")
    atomic_json(samples, root / "samples.json")
    atomic_json(diagnostics, root / "diagnostics.json")
    pd.DataFrame(rows).to_csv(root / "benchmark.csv", index=False)
    pd.DataFrame(ops).to_csv(root / "operator_counts.csv", index=False)
    del model, optimizer, state, batch
    gc.collect()
    torch.cuda.empty_cache()
    return rows, ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/cky_v25")
    parser.add_argument("--cache-dir", default="cache/tinystories_integration")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA PyTorch required")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 unsupported on this GPU; use --precision fp32")
    if args.repeats < 5 or args.warmup < 2:
        parser.error("Require >=5 samples and >=2 warmup")
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(dict(environment=device_info(), arguments=vars(args),
                     source_hashes={str(p): sha256(p) for p in Path("models").glob("*.py")},
                     protocol="Same model/AdamW/RNG snapshot restored before each sample; arm order alternates; "
                              "synchronized resident-input microbatch update, one optimizer step per batch. "
                              "Staged diagnostics and profiler counts are separate untimed runs."), output / "benchmark_environment.json")
    all_rows, all_ops = [], []
    for case in CASES:
        if args.cases and case[0] not in args.cases:
            continue
        print(f"Paired benchmark: {case[0]}", flush=True)
        rows, ops = run_case(case, args)
        all_rows.extend(rows)
        all_ops.extend(ops)
        pd.DataFrame(all_rows).to_csv(output / "benchmark.csv", index=False)
        pd.DataFrame(all_ops).to_csv(output / "operator_counts.csv", index=False)
        atomic_json({p.parent.name: __import__("json").loads(p.read_text(encoding="utf-8"))
                     for p in (output/"raw").glob("*/composer_workload.json")},output/"composer_workload.json")


if __name__ == "__main__":
    main()
