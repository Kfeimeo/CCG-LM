"""Sequential single-GPU seed/mode matrix and real module parameter counts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import pandas as pd
import torch
from models.model import LatentForestLM, MODES, parameter_report
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main_125m.yaml")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--output-dir", default="runs/matrix")
    parser.add_argument("--parameters-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.parameters_only:
        rows = []
        for mode in args.modes:
            mc = dict(cfg["model"], mode=mode)
            with torch.device("meta"):
                model = LatentForestLM(mc)
            report = parameter_report(model)
            rows.append(dict(config=cfg["name"], mode=mode, total=report["total"], **report["modules"]))
        pd.DataFrame(rows).fillna(0).to_csv(out / "parameters.csv", index=False)
        print(pd.DataFrame(rows).fillna(0).to_string(index=False))
        return
    for mode in args.modes:
        for seed in args.seeds:
            directory = out / f"{cfg['name']}_{mode}_seed{seed}"
            command = [sys.executable, "train.py", "--config", args.config, "--mode", mode, "--seed", str(seed),
                       "--output-dir", str(directory), "--device", args.device]
            if args.budget_tokens is not None:
                command += ["--budget-tokens", str(args.budget_tokens)]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
                if args.evaluate:
                    subprocess.run([sys.executable, "evaluate.py", "--checkpoints", *map(str, sorted(directory.glob("tokens_*.pt"))),
                                    "--cache-dir", cfg["data"]["cache_dir"], "--output-dir", str(directory / "evaluation"),
                                    "--device", args.device], check=True)
    if not args.dry_run:
        summaries = []
        for path in out.glob("*/metrics.csv"):
            metrics = pd.read_csv(path)
            run_cfg = json.loads((path.parent / "config.json").read_text())
            parameters = json.loads((path.parent / "parameters.json").read_text())
            row = dict(run=path.parent.name, mode=run_cfg["model"]["mode"], seed=run_cfg["seed"],
                       target_tokens=int(metrics.target_tokens.iloc[-1]), parameters=parameters["total"],
                       train_seconds=float(metrics.train_seconds.iloc[-1]),
                       target_tokens_per_second=float(metrics.step_tokens.sum() / metrics.step_seconds.sum()),
                       peak_gpu_gib=float(metrics.peak_gpu_gib.max()) if metrics.peak_gpu_gib.notna().any() else None)
            evaluated = path.parent / "evaluation" / "summary.csv"
            if evaluated.exists():
                frame = pd.read_csv(evaluated).sort_values("trained_target_tokens")
                row.update(final_dev_nll=float(frame.nll.iloc[-1]))
            summaries.append(row)
        pd.DataFrame(summaries).to_csv(out / "matrix_summary.csv", index=False)


if __name__ == "__main__":
    main()
