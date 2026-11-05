"""V2/V2.5 span, LM, gradient and training-trajectory equivalence checks."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path

import argparse
import copy
import itertools
from pathlib import Path
import torch
from torch.nn import functional as F

from models.composer import Composer
from models.decoder import Decoder
from models.forest import SemanticForest
from models.model import LatentForestLM
from models.reader import SemanticReader
from train import make_optimizer
from utils import atomic_json, autocast_context, load_config, restore_rng, rng_state, set_seed


def small_config(mode="forest_vector4", words=24):
    cfg = load_config("configs/smoke.yaml")
    cfg["model"].update(mode=mode, vocab_size=64, word_vocab_size=80, hidden_size=32,
                        layers=2, heads=4, ffn_size=64, dropout=0.1, cross_layers=[1, 2],
                        syntax_embedding=12, semantic_size=16, syntax_hidden=24,
                        syntax_layers=1, syntax_heads=4, composer_hidden=32,
                        coefficient_size=8, reader_heads=4, reader_width=32,
                        max_words=words, max_target_length=128, exploration=0.35)
    return cfg


def assert_tensor(reference, actual, label, bf16=False, exact=False, semantic=False):
    a, b = reference.detach().cpu(), actual.detach().cpu()
    atol, rtol = (0.04, 0.04) if bf16 else (3e-6, 3e-5)
    if bf16 and not exact:
        normalized=float((a.float()-b.float()).norm())/max(float(a.float().norm()),1e-30)
        assert normalized <= .02, f"{label}: normalized L2 error {normalized} > .02"
    try:
        torch.testing.assert_close(a, b, atol=0 if exact else atol, rtol=0 if exact else rtol)
    except AssertionError as error:
        if a.shape == b.shape and a.numel():
            mismatch = a.ne(b) if exact else ~torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol)
            positions = mismatch.nonzero()
            index = tuple(positions[0].tolist()) if len(positions) else ()
            detail = f"first index (batch, candidate, ...): {index}; ref={a[index]}; lazy={b[index]}"
        else:
            detail = f"shapes: ref={a.shape}, lazy={b.shape}"
        raise AssertionError(f"{label}: {detail}\n{error}") from None
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def compare_forests(a, b, bf16=False):
    errors = {}
    assert a["spans"].keys() == b["spans"].keys()
    for span in sorted(a["spans"], key=lambda s: (s[1] - s[0], s[0])):
        x, y = a["spans"][span], b["spans"][span]
        assert x["metadata"] == y["metadata"], f"Span {span}: candidate enumeration differs"
        for key in ("selected_flat", "retained_flat", "candidate_mask", "scores"):
            if x[key] is None:
                assert y[key] is None
            else:
                try:
                    error = assert_tensor(x[key], y[key], f"Span {span}, {key}", bf16,
                                          exact=key in ("selected_flat", "retained_flat", "candidate_mask", "scores"))
                    errors[key] = max(errors.get(key, 0), error)
                except AssertionError as error:
                    raise AssertionError(f"{error}\nReference identities: {x['retained_flat']}\nV2 identities: {y['retained_flat']}\nSources: {x['metadata']}") from None
    for (i,j),cell in b["spans"].items():
        if j-i == 1: continue
        flat=cell["retained_flat"].detach().cpu()
        n,k=flat.shape
        expected=[]
        for off,l,r in cell["metadata"]:
            lc,rc=b["spans"][i,i+off],b["spans"][i+off,j]
            expected.append((lc["candidate_base"]+l,rc["candidate_base"]+r,
                             lc["scores"].shape[-1],rc["scores"].shape[-1]))
        decoded=torch.tensor(expected)[flat]
        rows=torch.arange(n)[:,None]
        assert torch.equal(cell["left_id"].detach().cpu(),decoded[...,0]+rows*decoded[...,2])
        assert torch.equal(cell["right_id"].detach().cpu(),decoded[...,1]+rows*decoded[...,3])
    for key in ("semantic", "scores", "vector_coeff", "candidate_mask", "log_prior"):
        errors["root_"+key] = assert_tensor(a[key], b[key], f"Root {key}", bf16, exact=key in ("scores", "candidate_mask", "log_prior"), semantic=key=="semantic")
    for key in ("semantic","vector_coeff","log_prior"):
        x,y=a[key].detach().float(),b[key].detach().float()
        norm=float(x.norm())
        normalized=float((x-y).norm())/max(norm,1e-30)
        errors["root_"+key+"_reference_norm"]=norm
        errors["root_"+key+"_normalized_l2"]=normalized
        if key=="semantic": assert normalized <= (.02 if bf16 else 3e-5),normalized
    return errors


def forest_case(batch, words, k, r, vector, device="cpu", precision="fp32", policy="normal"):
    set_seed(1000 + batch * 100 + words * 10 + k + r)
    device = torch.device(device)
    bf16 = precision == "bf16"
    n, d, c = batch * 2, 16, 8 if vector else 0
    leaves = torch.randn(n, words, d, device=device)
    scores = torch.randn(n, words, words, device=device) * 0.15
    coeff = torch.randn(n, words, words, c, device=device) * 0.1
    lengths = ([words, max(1, words - 1), words // 2, 1, 0, words, 2, 3] * 2)[:n]
    lengths = [min(words,x) for x in lengths]
    mask = torch.arange(words, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    delta, exploration, training = 8.0, 0.0, False
    if policy == "explore":
        exploration, training = 1.0, True
    elif policy == "prune":
        delta = 0.0
    elif policy == "ties":
        scores.zero_()
    elif policy == "unreachable":
        for row, length in enumerate(lengths):
            if length:
                scores[row, 0, length - 1] = -torch.inf
        scores[-1].fill_(-torch.inf)
    elif policy == "empty":
        mask.zero_()
    elif policy == "partial_inf":
        scores[:, 0, 1] = -torch.inf
    ref = SemanticForest(Composer(d, 32), k, r, delta, exploration, "score_first").to(device).train(training)
    opt = SemanticForest(Composer(d, 32), k, r, delta, exploration, "lazy_semantic").to(device).train(training)
    opt.load_state_dict(ref.state_dict())
    state = rng_state()
    with torch.no_grad(), autocast_context(device, precision):
        a = ref(leaves, scores, coeff, mask, return_spans=True, return_trees=True)
    end_rng = rng_state()
    restore_rng(state)
    with torch.no_grad(), autocast_context(device, precision):
        b = opt(leaves, scores, coeff, mask, return_spans=True, return_trees=True)
    end_lazy = rng_state()
    assert torch.equal(end_rng["torch"], end_lazy["torch"])
    if device.type == "cuda":
        assert all(torch.equal(x,y) for x,y in zip(end_rng["cuda"],end_lazy["cuda"]))
    assert a["trees"] == b["trees"]
    errors = compare_forests(a, b, bf16)
    workload = opt.last_workload.copy()
    assert workload["v25_actual_composer_rows"] <= workload["v25_required_node_occurrences"]
    assert workload["v25_actual_composer_rows"] <= workload["v2_composer_candidate_rows"]
    assert all(torch.isfinite(b[key]).all() for key in ("semantic", "scores", "vector_coeff", "log_prior"))
    if policy in ("unreachable", "empty"):
        assert not b["candidate_mask"].any(), "Unreachable roots must remain explicitly invalid"
    if policy == "partial_inf":
        assert b["candidate_mask"][0].any(), "Reachable alternatives must survive -inf subspans"
        assert not b["spans"][0, 2]["candidate_mask"].any()
    if policy == "ties" and k > 1:
        assert b["candidate_mask"][0].sum() == k
        assert not torch.equal(b["semantic"][0, 0], b["semantic"][0, 1]), "Equal scores must not merge different semantics"
    cfg = small_config()["model"]
    cfg.update(semantic_size=d, coefficient_size=c, dropout=0.0)
    reader, decoder = SemanticReader(cfg, vector).to(device).eval(), Decoder(cfg).to(device).eval()
    readers = torch.nn.ModuleDict({"1": reader})
    def memory(result):
        result = {key: result[key].reshape(batch, 2, *result[key].shape[1:])
                  for key in ("semantic", "log_prior", "vector_coeff", "candidate_mask")}
        result["history_mask"] = mask.any(-1).reshape(batch, 2)
        return result
    ma, mb = memory(a), memory(b)
    query = torch.randn(batch, 5, cfg["hidden_size"], device=device)
    ids = torch.randint(4, cfg["vocab_size"], (batch, 5), device=device)
    labels = torch.randint(4, cfg["vocab_size"], (batch, 5), device=device)
    with torch.no_grad(), autocast_context(device, precision):
        ra = reader(query, ma)[0]
        rb = reader(query, mb)[0]
        la = decoder(ids, ma, readers)[0]
        lb = decoder(ids, mb, readers)[0]
        loss_a = F.cross_entropy(la.float().flatten(0, 1), labels.flatten())
        loss_b = F.cross_entropy(lb.float().flatten(0, 1), labels.flatten())
    for label, x, y in [("Reader", ra, rb), ("LM logits", la, lb), ("LM loss", loss_a, loss_b)]:
        errors[label] = assert_tensor(x, y, label, bf16)
    return dict(batch=batch, histories=2, words=words, k=k, r=r, vector=vector,
                device=str(device), precision=precision, policy=policy, spans=len(a["spans"]),
                loss_reference=float(loss_a), loss_lazy=float(loss_b), workload=workload, max_absolute_errors=errors, status="passed")


def model_batch(cfg, batch=2, words=8, histories=2, target_length=8, device="cpu"):
    mc = cfg["model"]
    ids = torch.randint(4, mc["vocab_size"], (batch, target_length), device=device)
    history = torch.randint(4, mc["word_vocab_size"], (batch, histories, words), device=device)
    mask = torch.ones_like(history, dtype=torch.bool)
    if batch > 1:
        mask[-1, -1, words // 2:] = False
    history = history.masked_fill(~mask, 0)
    labels = torch.cat((ids[:, 1:], torch.full((batch, 1), 3, device=device, dtype=torch.long)), -1)
    ids[:, 0] = 2
    return dict(input_ids=ids, labels=labels, history_ids=history, word_mask=mask, history_mask=mask.any(-1))


def gradient_statistics(a, b):
    maximum, relative, significant = 0.0, 0.0, 0.0
    norm_a = norm_b = squared_error = 0.0
    for name in a:
        x, y = a[name], b[name]
        if x is None or y is None:
            assert x is None and y is None, f"Gradient reachability differs: {name}"
            continue
        x, y = x.float(), y.float()
        diff = (x - y).abs()
        denominator = torch.maximum(x.abs(), y.abs())
        maximum = max(maximum, float(diff.max()))
        relative = max(relative, float((diff / denominator.clamp_min(1e-8)).max()))
        active = denominator >= 1e-5
        if active.any():
            significant = max(significant, float((diff[active] / denominator[active]).max()))
        norm_a += float(x.double().square().sum())
        norm_b += float(y.double().square().sum())
        squared_error += float((x.double() - y.double()).square().sum())
    return dict(max_absolute_gradient_error=maximum, max_relative_gradient_error=relative,
                max_relative_gradient_error_above_1e_5=significant,
                gradient_norm_reference=norm_a**0.5, gradient_norm_lazy=norm_b**0.5,
                normalized_l2_gradient_error=(squared_error / max(norm_a, 1e-30))**0.5)


def model_gradient_case(mode="forest_vector4", device="cpu", precision="fp32", main=False, strict=True):
    set_seed(42)
    cfg = load_config("configs/main_125m.yaml") if main else small_config(mode)
    cfg["model"]["mode"] = mode
    model = LatentForestLM(cfg["model"]).to(device).train()
    model.forest.capture_spans = True
    batch = model_batch(cfg, batch=1 if main else 2, words=24 if main else 8,
                        histories=4 if main else 2, target_length=16 if main else 8, device=device)
    snapshots, outputs = {}, {}
    initial = rng_state()
    for implementation in ("score_first", "lazy_semantic"):
        model.forest.implementation = implementation
        model.zero_grad(set_to_none=True)
        restore_rng(initial)
        with autocast_context(torch.device(device), precision):
            result = model(**batch)
        end_state=rng_state()
        if implementation == "score_first":
            first_rng=end_state
        else:
            assert torch.equal(first_rng["torch"],end_state["torch"])
            assert all(torch.equal(x,y) for x,y in zip(first_rng["cuda"],end_state["cuda"]))
        # Forward comparisons happen before backward releases this graph.
        if implementation == "score_first":
            first = result
        else:
            forward_errors = compare_forests(first["memory"], result["memory"], precision == "bf16")
            forward_errors["lm_logits_max_absolute_error"] = assert_tensor(first["logits"], result["logits"], "Full model logits", precision == "bf16")
            forward_errors["lm_logits_normalized_l2"] = float((first["logits"].detach().float()-result["logits"].detach().float()).norm()/first["logits"].detach().float().norm())
            forward_errors["lm_loss_max_absolute_error"] = assert_tensor(first["loss"], result["loss"], "Full model loss", precision == "bf16")
        result["loss"].backward()
        snapshots[implementation] = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                                     for name, p in model.named_parameters()}
        outputs[implementation] = float(result["loss"].detach())
    groups = {
        "syntax_encoder": lambda n: n.startswith("syntax.") and not n.startswith("syntax.span_scorer."),
        "span_scorer": lambda n: n.startswith("syntax.span_scorer."),
        "semantic_embedding": lambda n: n.startswith("semantic_embedding."),
        "composer": lambda n: n.startswith("forest.composer."),
        "vector_coefficient_head": lambda n: n.startswith("syntax.span_scorer.vector."),
        "semantic_reader": lambda n: n.startswith("readers."),
        "decoder": lambda n: n.startswith("decoder."),
    }
    reports,violations = {},[]
    for group, predicate in groups.items():
        a = {n: g for n, g in snapshots["score_first"].items() if predicate(n)}
        b = {n: g for n, g in snapshots["lazy_semantic"].items() if predicate(n)}
        if not a:
            reports[group] = {"status": "not present in scalar mode"}
            continue
        reports[group] = gradient_statistics(a, b)
        if reports[group]["normalized_l2_gradient_error"] > (.05 if precision == "bf16" else 3e-5):
            violations.append(dict(group=group,kind="normalized_l2"))
        reports[group]["parameters"] = {name: gradient_statistics({name:g},{name:b[name]}) for name,g in a.items()}
        for name, gradient in a.items():
            if gradient is not None:
                # Near-zero entries use absolute tolerance; relative errors
                # are still reported separately instead of hidden by the floor.
                atol, rtol = (0.00015, 0.04) if precision == "bf16" else (3e-6, 0.0003)
                try:
                    torch.testing.assert_close(gradient, b[name], atol=atol, rtol=rtol, msg=f"Gradient {name}")
                except AssertionError:
                    violations.append(dict(parameter=name,kind="elementwise",atol=atol,rtol=rtol))
    if strict and violations: raise AssertionError(violations)
    return dict(mode=mode, device=device, precision=precision, config="main_125m" if main else "tiny",
                losses=outputs, forward_errors=forward_errors, groups=reports, violations=violations,
                status="strict_tolerance_mismatch" if violations else "passed")


def training_case(mode="forest_vector4", device="cpu", steps=8, precision="fp32"):
    set_seed(2026)
    cfg = small_config(mode)
    a,b = LatentForestLM(cfg["model"]).to(device),LatentForestLM(cfg["model"]).to(device)
    b.load_state_dict(a.state_dict())
    a.forest.implementation,b.forest.implementation = "score_first","lazy_semantic"
    a.forest.capture_spans=b.forest.capture_spans=True
    oa,ob=make_optimizer(a,cfg["train"]),make_optimizer(b,cfg["train"])
    rows=[]
    for step in range(steps):
        batch=model_batch(cfg,device=device); state=rng_state(); results=[]; stats=[]
        for model,optimizer in ((a,oa),(b,ob)):
            restore_rng(state); optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch.device(device),precision): result=model(**batch)
            assert torch.isfinite(result["loss"])
            result["loss"].backward()
            norms={prefix:sum(float(p.grad.detach().float().square().sum()) for name,p in model.named_parameters()
                         if name.startswith(prefix) and p.grad is not None)**0.5
                   for prefix in ("syntax.span_scorer.","forest.composer.")}
            stats.append(dict(loss=float(result["loss"].detach()),gradient_norms=norms,
                              scalar_bias_gradient=float(model.syntax.span_scorer.scalar.bias.grad)))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True); optimizer.step()
            results.append(result)
        changed=0
        for span in results[0]["memory"]["spans"]:
            for key in ("selected_flat","retained_flat","candidate_mask"):
                x,y=results[0]["memory"]["spans"][span][key],results[1]["memory"]["spans"][span][key]
                if x is not None: changed += int(x.ne(y).sum())
        diffs={name:float((x.detach()-dict(b.named_parameters())[name].detach()).abs().max()) for name,x in a.named_parameters()}
        rows.append(dict(step=step+1,score_first=stats[0],lazy_semantic=stats[1],
            loss_error=abs(stats[0]["loss"]-stats[1]["loss"]),changed_search_entries=changed,
            max_parameter_error=max(diffs.values()),neutral_scalar_bias_error=diffs["syntax.span_scorer.scalar.bias"],
            max_non_gauge_parameter_error=max(v for k,v in diffs.items() if k!="syntax.span_scorer.scalar.bias")))
    return dict(mode=mode,device=device,precision=precision,steps=steps,trajectory=rows,
                final_parameter_errors=diffs,
                search_identities_unchanged=all(x["changed_search_entries"]==0 for x in rows),
                max_loss_error=max(x["loss_error"] for x in rows),status="finite; numerical drift recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/cky_v25")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--supplement-only",action="store_true",help="Keep saved matrix; rerun gradient and training diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    correctness = dict(status="running", matrix=[], edge_cases=[], training=[])
    if args.supplement_only:
        import json
        correctness=json.loads((output/"correctness.json").read_text(encoding="utf-8"))
        correctness["training"]=[]
    gradients = dict(status="running", cases=[],
                     metric_definitions=dict(relative_denominator="max(abs(reference), abs(lazy), 1e-8)",
                                             significant_threshold=1e-5, fp32_atol=3e-6, fp32_rtol=0.0003,
                                             bf16_atol=0.00015, bf16_rtol=0.04,
                                             bf16_forward_atol=.04,bf16_forward_rtol=.04,bf16_forward_normalized_l2_limit=.02,
                                             bf16_gradient_group_normalized_l2_limit=.05))
    devices = [("cpu", "fp32")]
    if not args.cpu_only and torch.cuda.is_available():
        devices.extend([("cuda", "fp32"), ("cuda", "bf16")])
    for device, precision in devices:
        if not args.supplement_only:
            for index, (batch, words, k, r, vector) in enumerate(itertools.product([1, 2, 4], [1, 4, 8, 16, 24], [1, 2, 4], [4, 12], [False, True])):
                for policy in ("normal", "explore"):
                    correctness["matrix"].append(forest_case(batch, words, k, r, vector, device, precision, policy))
                if index % 24 == 0:
                    print(f"Forward equivalence {device}/{precision}: {index + 1}/180", flush=True)
            for vector in (False, True):
                correctness["edge_cases"].append(forest_case(2, 1, 4, 12, vector, device, precision))
            for policy in ("normal", "prune", "explore", "ties", "unreachable", "empty", "partial_inf"):
                for vector in (False, True):
                    correctness["edge_cases"].append(forest_case(2, 8, 4, 12, vector, device, precision, policy))
        for mode in ("forest_scalar4", "forest_vector4"):
            gradients["cases"].append(model_gradient_case(mode, device, precision, strict=False))
            if device == "cpu" or precision == "bf16":
                correctness["training"].append(training_case(mode, device, precision=precision, steps=8 if device == "cpu" else 20))
        atomic_json(correctness, output / "correctness.json")
        atomic_json(gradients, output / "gradient_comparison.json")
    if not args.cpu_only and not args.skip_main and torch.cuda.is_available():
        for precision in ("fp32", "bf16"):
            gradients["cases"].append(model_gradient_case("forest_vector4", "cuda", precision, main=True, strict=False))
            print(f"Main model gradients {precision}: {gradients['cases'][-1]['status']}", flush=True)
    correctness["status"] = "search_and_small_model_forward_passed"
    correctness["training_status"] = "fp32_losses_and_search_consistent_bf16_trajectories_diverge"
    gradients["status"] = "passed" if all(c["status"]=="passed" for c in gradients["cases"]) else "strict_tolerance_mismatch"
    atomic_json(correctness, output / "correctness.json")
    atomic_json(gradients, output / "gradient_comparison.json")
    print(f"Search/forward: {len(correctness['matrix'])} matrix cases, {len(correctness['edge_cases'])} edges; "
          f"gradients: {gradients['status']} ({len(gradients['cases'])} cases); "
          f"{len(correctness['training'])} training trajectories with drift recorded")
    if gradients["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
