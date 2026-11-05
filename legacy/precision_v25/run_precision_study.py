"""Reproduce V2.5-P on one GPU; use a different output folder for each GPU."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse,subprocess,sys,json
from pathlib import Path
from utils import atomic_json,device_info
HERE=Path(__file__).resolve().parent  # legacy/precision_v25; relative output/cache paths still resolve from the repo root


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="reports/cky_v25_precision")
    p.add_argument("--cache-dir",default="cache/tinystories_integration");p.add_argument("--resume",action="store_true")
    p.add_argument("--post-benchmark",action="store_true",help="Numerical and performance phases already complete")
    a=p.parse_args();root=Path(a.output_dir);logs=root/"logs";logs.mkdir(parents=True,exist_ok=True)
    environment=device_info()
    if (root/"run_environment.json").exists():
        old=json.loads((root/"run_environment.json").read_text(encoding="utf-8"))["environment"]
        if any(old.get(k)!=environment.get(k) for k in ("gpu","torch","cuda_build")):
            raise ValueError("GPU/runtime changed: use a separate output directory")
    atomic_json(dict(environment=environment,python=sys.version,arguments=vars(a)),root/"run_environment.json")
    def run(script,args,name):
        print(name,flush=True)
        with (logs/(name+".log")).open("w",encoding="utf-8") as f:
            r=subprocess.run([sys.executable,str(script),*args],stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(name+" failed; see log")
    if not a.post_benchmark:
        run(HERE/"validate_precision_v25.py",["--output-dir",str(root)]+(["--resume"] if a.resume else []),"numerical")
        run(HERE/"benchmark_precision_v25.py",["--output-dir",str(root/"performance"),"--cache-dir",a.cache_dir]+(["--resume"] if a.resume else []),"performance")
    # On another GPU, regenerate audit metadata from that run's actual hooks.
    if not (root/"feasibility.json").exists():
        measured=json.loads((root/"numerical_A.json").read_text(encoding="utf-8"))
        atomic_json(dict(environment=environment,backend=measured["backend"],
                         measured_dtypes=measured["models"][-1]["dtypes"],
                         audit="Parameter FP32 and operator dtypes are measured, not inferred from policy labels"),root/"feasibility.json")
    import shutil
    shutil.copyfile(Path(a.cache_dir)/"manifest.json",root/"dataset_manifest.json")
    for group in "ABCDE":run(HERE/"validate_precision_batch4.py",["--group",group,"--output-dir",str(root)],"backward_batch4_"+group)
    for group in ("A","B"):run(HERE/"validate_precision_controls.py",["--group",group,"--output-dir",str(root)],"controls_"+group)
    for mode in ("score","E","D","compare"):run(HERE/"precision_supplement.py",["--mode",mode,"--output-dir",str(root)],"supplement_"+mode)
    print("Regression suite",flush=True)
    with (logs/"pytest.log").open("w",encoding="utf-8") as f:
        r=subprocess.run([sys.executable,"-m","pytest","-q","--tb=short","--basetemp=.pytest_tmp_precision_release","-o","cache_dir=.pytest_cache_precision_release"],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError("Regression suite failed; see log")
    run(HERE/"train_precision_stability.py",["--output-dir",str(root/"stability_runs"),"--cache-dir",a.cache_dir]+(["--resume"] if a.resume else []),"stability")
    run(HERE/"summarize_precision_study.py",["--output-dir",str(root)],"report")

if __name__=="__main__":main()
