"""Reproducible CUDA micro-batch timings and optional CKY operator traces.

Uninstrumented timings measure complete forward/backward/clip/AdamW steps after
warmup. A separate diagnostic cuts the autograd graph at Syntax/Forest outputs
so decoder, CKY and syntax backward costs can be measured individually. These
synchronized diagnostic timings must not be added to uninstrumented throughput.
No checkpoint is saved and no benchmark text is treated as a research result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import cProfile
import gc
import json
from pathlib import Path
import pstats
import statistics
import time

import pandas as pd
import torch

from data.dataset import Collator, StoryDataset
from models.model import LatentForestLM, MODES, parameter_report
from train import make_optimizer
from utils import atomic_json, autocast_context, device_info, load_config, set_seed, sha256, to_device


def make_batch(cfg, batch_size, histories, words, target_length, source):
    mc = cfg["model"]
    if source == "synthetic":
        labels = torch.randint(4, mc["vocab_size"], (batch_size, target_length))
        labels[:, -1] = 3
        ids = torch.cat((torch.full((batch_size, 1), 2), labels[:, :-1]), -1)
        history = torch.randint(4, mc["word_vocab_size"], (batch_size, histories, words))
        return dict(input_ids=ids, labels=labels, target_mask=labels.ne(-100), history_ids=history,
                    word_mask=history.ne(0), history_mask=history.ne(0).any(-1))
    dataset = StoryDataset(cfg["data"]["cache_dir"])
    examples = []
    # Choose one full-history target per story; avoid accidentally benchmarking
    # only empty-history first sentences in an unshuffled batch.
    for start, end in zip(dataset.story_offsets[:-1], dataset.story_offsets[1:]):
        if end - start <= histories:
            continue
        ex = dataset[int(start + histories)]
        ex["history"] = [sentence[:words] for sentence in ex["history"][-histories:]]
        ex["input_ids"], ex["labels"] = ex["input_ids"][:target_length], ex["labels"][:target_length]
        examples.append(ex)
        if len(examples) == batch_size:
            return Collator()(examples)
    raise ValueError("Not enough stories with completed histories for this batch size")


def regular_step(model, batch, optimizer, precision):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(torch.device("cuda"), precision):
        result = model(**batch)
    result["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def time_regular(model, batch, optimizer, precision, warmup, repeats):
    for _ in range(warmup):
        regular_step(model, batch, optimizer, precision)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        start.record()
        regular_step(model, batch, optimizer, precision)
        end.record()
        torch.cuda.synchronize()
        samples.append(dict(wall_ms=(time.perf_counter() - tick) * 1000,
                            cuda_envelope_ms=start.elapsed_time(end)))
    return samples, torch.cuda.max_memory_allocated() / 2**30


class Diagnostics:
    def __init__(self):
        self.records = []

    def start(self, name):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        record = dict(name=name, a=a, b=b, tick=time.perf_counter())
        a.record()
        record["range"] = torch.profiler.record_function(name)
        record["range"].__enter__()
        return record

    def end(self, record):
        record["range"].__exit__(None, None, None)
        record["b"].record()
        record["cpu_submit_ms"] = (time.perf_counter() - record["tick"]) * 1000
        self.records.append(record)

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        record = self.start(name)
        try:
            yield
        finally:
            self.end(record)
            torch.cuda.synchronize()
            record["wall_ms"] = (time.perf_counter() - record["tick"]) * 1000

    def hook(self, module, name):
        stack = []
        def before(module, args):
            stack.append(self.start(name))
        def after(module, args, result):
            self.end(stack.pop())
        return [module.register_forward_pre_hook(before), module.register_forward_hook(after)]

    def finish(self):
        torch.cuda.synchronize()
        totals = defaultdict(lambda: dict(cpu_submit_ms=0.0, cuda_envelope_ms=0.0, calls=0))
        for row in self.records:
            dest = totals[row["name"]]
            dest["cpu_submit_ms"] += row["cpu_submit_ms"]
            dest["cuda_envelope_ms"] += row["a"].elapsed_time(row["b"])
            dest["calls"] += 1
            if "wall_ms" in row:
                dest["wall_ms"] = dest.get("wall_ms", 0) + row["wall_ms"]
        return dict(totals)


def staged_step(model, batch, optimizer, precision, scaler=None, measure_composer_backward=False, measure_reader_backward=False):
    """Detach three graph boundaries and propagate their gradients in stages.

    Only the diagnostic uses these hooks; the real training implementation and
    uninstrumented benchmark keep the usual single backward call.
    """
    timer, handles, boundaries = Diagnostics(), [], defaultdict(list)
    def cut_tensor(value, name):
        cut = value.detach().requires_grad_(value.requires_grad)
        boundaries[name].append((value, cut))
        return cut
    def cut_syntax(module, args, output):
        return tuple(cut_tensor(value, "syntax") for value in output)
    def cut_embedding(module, args, output):
        return cut_tensor(output, "embedding")
    def cut_forest(module, args, output):
        result = dict(output)
        for key in ("semantic", "log_prior", "vector_coeff"):
            result[key] = cut_tensor(output[key], "forest")
        return result
    def backprop(name):
        pairs = [(original, cut.grad) for original, cut in boundaries[name] if cut.grad is not None and original.requires_grad]
        if pairs:
            torch.autograd.backward([x[0] for x in pairs], [x[1] for x in pairs])
    for module, name in [(model.syntax, "module/syntax"), (model.forest, "module/cky"),
                         (model.composer, "module/composer"), (model.decoder, "module/decoder")]:
        if module is not None:
            handles.extend(timer.hook(module, name))
    for reader in model.readers.values():
        handles.extend(timer.hook(reader, "module/reader"))
    if model.syntax is not None:
        handles.append(model.syntax.register_forward_hook(cut_syntax))
    if model.semantic_embedding is not None:
        handles.append(model.semantic_embedding.register_forward_hook(cut_embedding))
    if model.forest is not None:
        handles.append(model.forest.register_forward_hook(cut_forest))
    if measure_composer_backward:
        backward_stack=[]
        handles.append(model.composer.register_full_backward_pre_hook(lambda m,g:backward_stack.append(timer.start("module/composer_backward"))))
        handles.append(model.composer.register_full_backward_hook(lambda m,gi,go:timer.end(backward_stack.pop())))
    if measure_reader_backward:
        reader_stacks={id(r):[] for r in model.readers.values()}
        def reader_before(m,g):reader_stacks[id(m)].append(timer.start("module/reader_backward"))
        def reader_after(m,gi,go):timer.end(reader_stacks[id(m)].pop())
        for r in model.readers.values():
            handles.append(r.register_full_backward_pre_hook(reader_before))
            handles.append(r.register_full_backward_hook(reader_after))
    optimizer.zero_grad(set_to_none=True)
    try:
        with timer.phase("phase/forward"):
            with autocast_context(torch.device("cuda"), precision):
                result = model(**batch)
        with timer.phase("phase/decoder_backward"):
            (scaler.scale(result["loss"]) if scaler is not None else result["loss"]).backward()
        if boundaries["forest"]:
            with timer.phase("phase/cky_backward"):
                backprop("forest")
        if boundaries["syntax"] or boundaries["embedding"]:
            with timer.phase("phase/syntax_embedding_backward"):
                # The two independent vocabulary channels share no graph.
                backprop("syntax")
                backprop("embedding")
        with timer.phase("phase/clip_optimizer"):
            if scaler is not None and scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if scaler is not None: scaler.step(optimizer);scaler.update()
            else: optimizer.step()
        return timer.finish()
    finally:
        for handle in handles:
            handle.remove()


def export_trace(model, batch, optimizer, precision, output):
    output.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True, with_stack=False) as prof:
        staged_step(model, batch, optimizer, precision)
    prof.export_chrome_trace(str(output / "trace.json"))
    averages = prof.key_averages()
    (output / "operators.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=45), encoding="utf-8")
    rows = [dict(operator=e.key, calls=e.count, self_cpu_ms=e.self_cpu_time_total / 1000,
                 cpu_total_ms=e.cpu_time_total / 1000, self_device_ms=e.self_device_time_total / 1000,
                 device_total_ms=e.device_time_total / 1000) for e in averages]
    pd.DataFrame(rows).to_csv(output / "operators.csv", index=False)
    phase_ops = defaultdict(lambda: dict(calls=0, self_cpu_ms=0.0, self_device_ms=0.0))
    for event in prof.events():
        parent, phase = event.cpu_parent, None
        while parent is not None:
            if parent.name in ("module/cky", "module/syntax", "module/decoder", "phase/cky_backward",
                               "phase/syntax_embedding_backward", "phase/decoder_backward", "phase/clip_optimizer"):
                phase = parent.name
                break
            parent = parent.cpu_parent
        if phase and event.device_type == torch.autograd.DeviceType.CPU:
            row = phase_ops[phase, event.name]
            row["calls"] += 1
            row["self_cpu_ms"] += event.self_cpu_time_total / 1000
            row["self_device_ms"] += event.self_device_time_total / 1000
    pd.DataFrame([dict(phase=phase, operator=op, **row) for (phase, op), row in phase_ops.items()]).to_csv(output / "phase_operators.csv", index=False)
    host = cProfile.Profile()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    host.enable()
    with autocast_context(torch.device("cuda"), precision):
        result = model(**batch)
    host.disable()
    torch.cuda.synchronize()
    with (output / "python_forward.txt").open("w", encoding="utf-8") as handle:
        pstats.Stats(host, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(55)
    del result
    return [row for row in rows if row["operator"].startswith(("module/", "phase/"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["no_memory", "forest_vector4"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--source", choices=["synthetic", "cache"], default="synthetic")
    parser.add_argument("--cache-dir")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--forest-implementation", choices=["reference", "score_first", "lazy_semantic", "lazy_tensorized_a", "lazy_tensorized"])
    parser.add_argument("--output-dir", default="runs/cuda_profile")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA PyTorch; run with the CUDA Python environment")
    if min(args.batch_size, args.histories, args.target_length, args.repeats, *args.lengths) <= 0 or args.warmup < 1:
        parser.error("Sizes/repeats must be positive and warmup >=1")
    cfg = load_config(args.config)
    if args.forest_implementation is not None:
        cfg["model"]["forest_implementation"] = args.forest_implementation
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 not supported; use --precision fp32")
    if args.histories > cfg["model"]["max_history"] or max(args.lengths) > cfg["model"]["max_words"] or args.target_length > cfg["model"]["max_target_length"]:
        parser.error("Requested shape exceeds model capacities")
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_files = [Path("train.py"), *Path("models").glob("*.py")]
    atomic_json(dict(environment=device_info(), arguments=vars(args),
                     source_hashes={str(p): sha256(p) for p in source_files},
                     notes="Resident inputs; one micro-batch per optimizer update, no accumulation, validation or checkpoint IO. "
                           "Synthetic histories are all valid; cache mode uses true story-separated inputs. CUDA event envelopes include launch gaps; "
                           "operator device sums and instrumented diagnostics are not uninstrumented wall-clock throughput."), output / "environment.json")
    summary, details = [], []
    for mode in args.modes:
        set_seed(cfg["seed"])
        model = LatentForestLM(dict(cfg["model"], mode=mode), cfg["seed"]).cuda().train()
        optimizer = make_optimizer(model, cfg["train"])
        for words in (args.lengths[-1:] if mode == "no_memory" else args.lengths):
            batch = to_device(make_batch(cfg, args.batch_size, args.histories, words, args.target_length, args.source), torch.device("cuda"))
            case = f"{mode}_b{args.batch_size}_w{words}"
            try:
                samples, peak = time_regular(model, batch, optimizer, args.precision, args.warmup, args.repeats)
                wall = statistics.median(row["wall_ms"] for row in samples)
                row = dict(case=case, mode=mode, history_words=words, batch_size=args.batch_size,
                           forest_implementation=model.forest.implementation if model.forest is not None else "none",
                           target_length=int(batch["input_ids"].shape[1]), history_shape=list(batch["history_ids"].shape),
                           target_tokens=int(batch["labels"].ne(-100).sum()), parameters=parameter_report(model)["total"],
                           median_step_ms=wall, min_step_ms=min(s["wall_ms"] for s in samples),
                           max_step_ms=max(s["wall_ms"] for s in samples),
                           median_cuda_envelope_ms=statistics.median(s["cuda_envelope_ms"] for s in samples),
                           peak_allocated_gib=peak)
                row["target_tokens_per_second"] = row["target_tokens"] * 1000 / wall
                diagnostics = [staged_step(model, batch, optimizer, args.precision) for _ in range(min(3, args.repeats))]
                for name in diagnostics[0]:
                    details.append(dict(case=case, component=name,
                                        **{key: statistics.median(d[name][key] for d in diagnostics) for key in diagnostics[0][name]}))
                if args.trace and model.forest is not None and words == args.lengths[-1]:
                    ranges = export_trace(model, batch, optimizer, args.precision, output / case)
                    atomic_json(ranges, output / case / "profiler_ranges.json")
                summary.append(row)
                atomic_json(samples, output / f"{case}_samples.json")
                print(json.dumps(row), flush=True)
            except torch.OutOfMemoryError:
                raise RuntimeError(f"CUDA OOM for {case}; reduce --batch-size (try 1) or target length") from None
            finally:
                optimizer.zero_grad(set_to_none=True)
                del batch
                gc.collect()
                torch.cuda.empty_cache()
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(details).to_csv(output / "components.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
