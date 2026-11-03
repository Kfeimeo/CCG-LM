"""Generate the portable, Git-ignored Train.ipynb launcher (stdlib only)."""
import argparse
import json
from pathlib import Path
import textwrap
import uuid

CELLS = [
    ('markdown', r'''
# Latent Semantic Forest LM — 独立训练入口

本文件可以放在仓库外，单独上传到 Colab；不需要在侧栏看到其他文件。按顺序执行：设置 → 克隆仓库 → 安装依赖 → HF认证 → 数据 → 训练。
Colab先在“运行时 → 更改运行时类型”选择GPU；本机选择已有的py312 CUDA内核。

默认克隆 `https://github.com/Kfeimeo/CCG-LM.git` 的 `main`，已有源码目录则直接复用，不自动pull或覆盖本地修改。
GitHub只能克隆已提交且已推送的版本；若使用本机新增的CNN等功能，先将对应代码推送到所选分支。

默认 V3 smoke、32个真实训练故事的小缓存、512个CE目标Token，用于检查流程，不代表正式预训练。
本Notebook仅调用仓库的 `train.run()`，保留tqdm、精度策略、优化器、检查点恢复与开发集评估。

HF token通过Colab Secrets中名为 `HF_TOKEN` 的条目读取（授权本Notebook访问），或使用下面可选的隐藏输入。本文件不保存token明文。
本地 `Train.ipynb` 已设为Git忽略文件；分享前仍应清空执行输出。
'''),
    ('code', r'''
import copy, gc, importlib.util, json, os, shutil, subprocess, sys, uuid
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL = "https://github.com/Kfeimeo/CCG-LM.git"
REPO_REF = "main"  # 分支或tag；已有目录只复用，不自动切换分支
WORKSPACE_DIR = None  # Colab默认/content；本机默认当前目录
PROJECT_DIR = None  # 可填已有源码的绝对路径；None自动寻找或克隆
CLONE_IF_MISSING = True
INSTALL_DEPENDENCIES = IN_COLAB  # 本机py312通常不需要重新安装

USE_GOOGLE_DRIVE = False  # Colab需要持久化数据/检查点时设True
DRIVE_MOUNT_DIR = "/content/drive"
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/CCG-LM"
STORAGE_DIR = None  # 可指定数据和输出的根目录；与源码目录独立

HF_SECRET_NAME = "HF_TOKEN"
PROMPT_FOR_HF_TOKEN = False  # 没有Secret/环境变量/本地token文件时，可设True隐藏输入
'''),
    ('markdown', r'''
## 0. 克隆源码、安装依赖与认证

下方代码等价于 `git clone --depth 1 --branch main https://github.com/Kfeimeo/CCG-LM.git <PROJECT_DIR>`，使用参数列表执行，支持包含空格的路径。
本机可直接指定已有 `PROJECT_DIR`，无需复制Notebook到源码目录。已有目录不完整时会报错，避免覆盖文件。

Colab默认源码位于 `/content/CCG-LM`，数据和结果位于 `/content/ccg_lm_workspace`。
开启Google Drive后，源码仍在运行时本地，数据与结果默认位于 `DRIVE_PROJECT_DIR`，重连后可恢复。
'''),
    ('code', r'''
LAUNCH_DIR = Path.cwd().resolve()
WORKSPACE_ROOT = Path(WORKSPACE_DIR or ("/content" if IN_COLAB else LAUNCH_DIR)).expanduser().resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

def is_project(folder):
    return (folder / "train.py").is_file() and (folder / "models/model.py").is_file() and (folder / "configs/smoke.yaml").is_file()

if PROJECT_DIR:
    candidate = Path(PROJECT_DIR).expanduser()
    PROJECT_ROOT = (candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate).resolve()
else:
    existing = next((p for p in [LAUNCH_DIR, *LAUNCH_DIR.parents] if is_project(p)), None)
    PROJECT_ROOT = existing if existing else WORKSPACE_ROOT / "CCG-LM"

if not is_project(PROJECT_ROOT):
    if PROJECT_ROOT.exists() and (not PROJECT_ROOT.is_dir() or any(PROJECT_ROOT.iterdir())):
        raise FileExistsError(f"源码目录已有其他文件，请另选PROJECT_DIR: {PROJECT_ROOT}")
    if not CLONE_IF_MISSING:
        raise FileNotFoundError(f"找不到源码目录: {PROJECT_ROOT}")
    if shutil.which("git") is None:
        raise RuntimeError("运行环境未安装git，请先安装git或指定已有源码目录。")
    PROJECT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1", "--single-branch"]
    if REPO_REF:
        command += ["--branch", REPO_REF]
    subprocess.run(command + [REPO_URL, str(PROJECT_ROOT)], check=True)
    if not is_project(PROJECT_ROOT):
        raise RuntimeError("克隆完成，但该分支不是预期的CCG-LM项目。")

if USE_GOOGLE_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Google Drive挂载选项仅适用于Colab；本机直接设置STORAGE_DIR。")
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_DIR)

default_storage = Path(DRIVE_PROJECT_DIR) if USE_GOOGLE_DRIVE else (
    WORKSPACE_ROOT / "ccg_lm_workspace" if IN_COLAB else PROJECT_ROOT)
storage = Path(STORAGE_DIR).expanduser() if STORAGE_DIR else default_storage
STORAGE_ROOT = (storage if storage.is_absolute() else WORKSPACE_ROOT / storage).resolve()
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

def project_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()

def storage_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else STORAGE_ROOT / p).resolve()

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("源码目录:", PROJECT_ROOT)
print("数据和输出根目录:", STORAGE_ROOT)
'''),
    ('code', r'''
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements.txt"), "ipywidgets"], check=True)
else:
    print("复用当前环境；如缺少依赖，可将INSTALL_DEPENDENCIES设为True后运行本单元。")

import torch
from IPython.display import display
from utils import load_config, device_info, atomic_json

print("当前Python:", sys.executable)
print(json.dumps(device_info(), indent=2, ensure_ascii=False))
'''),
    ('code', r'''
from getpass import getpass

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
HF_TOKEN_SOURCE = "环境变量" if HF_TOKEN else None
if not HF_TOKEN and IN_COLAB:
    from google.colab import userdata
    try:
        HF_TOKEN = (userdata.get(HF_SECRET_NAME) or "").strip() or None
        HF_TOKEN_SOURCE = "Colab Secrets" if HF_TOKEN else None
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError, userdata.TimeoutException):
        print("未读取到Colab Secret；可设置HF_TOKEN并授权访问，或启用隐藏输入。")
if not HF_TOKEN:
    token_file = PROJECT_ROOT / ".hf_token"
    if token_file.is_file():
        HF_TOKEN = token_file.read_text(encoding="utf-8").strip() or None
        HF_TOKEN_SOURCE = "本机忽略的token文件" if HF_TOKEN else None
if not HF_TOKEN and PROMPT_FOR_HF_TOKEN:
    HF_TOKEN = getpass("Hugging Face token（输入隐藏，留空使用匿名访问）: ").strip() or None
    HF_TOKEN_SOURCE = "隐藏输入" if HF_TOKEN else None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN  # 仅保留在当前进程环境，不调用持久化login
print("HF认证来源:", HF_TOKEN_SOURCE or "匿名访问；已有缓存无需下载认证")
'''),
    ('markdown', r'''
## 1. 参数

- `CONFIG`：`configs/v3.yaml` 是 smoke；`configs/v3_main_125m.yaml` 是主模型；`configs/v3_fp32.yaml` 是全 FP32 主模型。
- `BUDGET_TOKENS` 是**累计总预算**，不是额外 Token；正式实验可自行提高。
- `EFFECTIVE_TOKENS` 是每次 optimizer update 的 CE Token 数，`MICRO_BATCH_SIZE` 影响显存和累积次数。
- `DATA_SOURCE`：`existing` 复用本地缓存；`demo` 生成合成缓存；`tinystories_subset` 下载小子集；`tinystories_full` 下载完整数据。
- 从已有检查点恢复时，使用其模型、Seed、精度、学习率和有效 Token 配置；下方 `RESUME_*` 只在显式填写时切换实现或 microbatch。

CNN baseline: set both `CONFIG="configs/ablations/seq_cnn.yaml"` and `MODE="seq_cnn4"`. For the kernel=1 control use `configs/ablations/seq_cnn_k1.yaml`. Details: [local CNN ablation](reports/local_cnn/README.md).

路径规则：`CONFIG`相对源码目录；`CACHE_DIR`、`RUN_DIR`、`RESUME_CHECKPOINT`相对数据和输出根目录；所有路径也都可以使用绝对路径。
'''),
    ('code', r'''
CONFIG = "configs/v3.yaml"
MODE = "forest_vector4"
FOREST_IMPLEMENTATION = None  # None 跟随配置，也可选 score_first / lazy_semantic / lazy_tensorized
SEED = 42
DEVICE = "cuda"  # 使用 CPU 时显式改为 "cpu"；也可用 "auto"

BUDGET_TOKENS = 512
EFFECTIVE_TOKENS = 256
MICRO_BATCH_SIZE = 2
SAVE_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 10
TRAIN_EVAL_BATCHES = 2  # 训练期开发集检查范围；不是完整开发集指标

DATA_SOURCE = "existing"
CACHE_DIR = None  # 相对STORAGE_ROOT或绝对路径；None使用默认缓存
VERIFY_CACHE = True

RESUME_CHECKPOINT = None  # 例如 r"runs/your_run/tokens_512.pt"；只加载可信检查点
RESUME_IMPLEMENTATION = None  # 例如 "lazy_tensorized"，None 保留检查点实现
RESUME_MICRO_BATCH_SIZE = None  # OOM 后可减小；None 保持原值
RUN_DIR = None  # 新训练自动建唯一目录；恢复默认使用检查点所在目录
'''),
    ('markdown', r'''
## 2. 生成本次配置

默认不改写仓库 YAML。解析后的配置写入本次运行目录，便于复现。
已有输出目录若含训练记录，新训练会拒绝覆盖；应改用检查点恢复或选择新目录。
'''),
    ('code', r'''
IMPLEMENTATIONS = {"reference", "score_first", "lazy_semantic", "lazy_tensorized_a", "lazy_tensorized", "lazy_tensorized_score_reconstruct"}
RESUME_PATH = storage_path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None
START_TOKENS = 0
if RESUME_PATH:
    if not RESUME_PATH.is_file():
        raise FileNotFoundError(RESUME_PATH)
    saved = torch.load(RESUME_PATH, map_location="cpu", weights_only=False)
    CFG = copy.deepcopy(saved["config"])
    START_TOKENS = int(saved["state"]["target_tokens"])
    if RESUME_IMPLEMENTATION is not None:
        CFG["model"]["forest_implementation"] = RESUME_IMPLEMENTATION
    if RESUME_MICRO_BATCH_SIZE is not None:
        CFG["train"]["micro_batch_size"] = RESUME_MICRO_BATCH_SIZE
    del saved
    print("恢复检查点配置；忽略新训练的 CONFIG / MODE / SEED / EFFECTIVE_TOKENS。")
else:
    config_file = project_path(CONFIG)
    if not config_file.is_file():
        raise FileNotFoundError(f"所选分支缺少配置 {CONFIG}；请检查REPO_REF或先推送对应代码。")
    CFG = load_config(config_file)
    CFG["model"]["mode"] = MODE
    if FOREST_IMPLEMENTATION is not None:
        CFG["model"]["forest_implementation"] = FOREST_IMPLEMENTATION
    CFG["seed"] = SEED
    CFG["train"].update(effective_tokens=EFFECTIVE_TOKENS, micro_batch_size=MICRO_BATCH_SIZE)

if CFG["model"].get("forest_implementation", "reference") not in IMPLEMENTATIONS:
    raise ValueError("未知的 CKY 实现")
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("当前内核没有CUDA：Colab请选择GPU运行时，本机请选择CUDA内核；CPU运行请设置DEVICE='cpu'。")
if min(BUDGET_TOKENS, CFG["train"]["effective_tokens"], CFG["train"]["micro_batch_size"]) <= 0:
    raise ValueError("Token 预算和 microbatch 必须为正数")
if BUDGET_TOKENS <= START_TOKENS:
    raise ValueError(f"检查点已有 {START_TOKENS} Token，请把总预算设得更大。")
if CFG.get("precision"):
    from models.precision_policy import PrecisionPolicy
    PrecisionPolicy(**CFG["precision"])  # 校验策略；实际训练入口负责设置后端
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        raise RuntimeError("当前精度配置需要支持 fp32_precision 的 PyTorch；请升级到支持该API的CUDA版PyTorch后重启内核，或选择兼容的精度配置。")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ROOT = storage_path(RUN_DIR) if RUN_DIR else (
    RESUME_PATH.parent if RESUME_PATH else STORAGE_ROOT / "runs" / f"notebook_{CFG['name']}_{stamp}_{uuid.uuid4().hex[:6]}"
)
if not RESUME_PATH and RUN_ROOT.exists() and ((RUN_ROOT / "metrics.csv").exists() or any(RUN_ROOT.glob("*.pt"))):
    raise FileExistsError("该目录已有训练结果，请设置 RESUME_CHECKPOINT 或更换 RUN_DIR。")

CFG["train"].update(budget_tokens=BUDGET_TOKENS, device=DEVICE, output_dir=str(RUN_ROOT), progress=True,
                    save_every_steps=SAVE_EVERY_STEPS, eval_every_steps=EVAL_EVERY_STEPS,
                    eval_batches=TRAIN_EVAL_BATCHES, drive_dir=None)
CFG["train"]["checkpoint_tokens"] = sorted(set(
    [int(x) for x in CFG["train"]["checkpoint_tokens"] if int(x) <= BUDGET_TOKENS] + [BUDGET_TOKENS]
))
print(json.dumps({"run": str(RUN_ROOT), "resume": str(RESUME_PATH) if RESUME_PATH else None,
                  "from_tokens": START_TOKENS, "total_budget": BUDGET_TOKENS,
                  "mode": CFG["model"]["mode"], "cky": CFG["model"].get("forest_implementation"),
                  "precision": CFG.get("precision", CFG["train"]["precision"]),
                  "effective_tokens": CFG["train"]["effective_tokens"], "micro_batch_size": CFG["train"]["micro_batch_size"]},
                 indent=2, ensure_ascii=False))
DATA_READY = False
'''),
    ('markdown', r'''
## 3. 数据准备与校验

默认在数据根目录复用 `cache/tinystories_integration`；不存在时，从源码仓库附带的小缓存归档解压到数据目录。只在显式选择TinyStories下载源时下载；认证使用上方HF_TOKEN。
'''),
    ('code', r'''
import zipfile
from data.prepare import prepare, verify_cache

sources = {"existing", "demo", "tinystories_subset", "tinystories_full"}
if DATA_SOURCE not in sources:
    raise ValueError(f"DATA_SOURCE 必须是 {sorted(sources)}")
if RESUME_PATH and DATA_SOURCE != "existing":
    raise ValueError("恢复训练使用existing；移动数据后请设置CACHE_DIR，保持原缓存指纹。")
default_cache = {
    "existing": "cache/tinystories_integration", "demo": "cache/notebook_demo",
    "tinystories_subset": "cache/tinystories_integration", "tinystories_full": "cache/tinystories_full",
}
chosen_cache = CACHE_DIR or (CFG["data"]["cache_dir"] if RESUME_PATH else default_cache[DATA_SOURCE])
CACHE_ROOT = storage_path(chosen_cache)
CFG["data"]["cache_dir"] = str(CACHE_ROOT)
if DATA_SOURCE == "existing":
    bundle = PROJECT_ROOT / "precision_study_cache.zip"
    if not (CACHE_ROOT / "manifest.json").exists() and CACHE_ROOT == storage_path(default_cache["existing"]) and bundle.is_file():
        # Archive paths are rooted at cache/tinystories_integration;
        # relocate contents to the independent storage directory.
        prefix = PurePosixPath("cache/tinystories_integration")
        with zipfile.ZipFile(bundle) as z:
            destinations = []
            for member in z.infolist():
                archive_path = PurePosixPath(member.filename)
                if chr(92) in member.filename or ".." in archive_path.parts or not archive_path.is_relative_to(prefix):
                    raise ValueError(f"缓存归档路径不合法: {member.filename}")
                relative = archive_path.relative_to(prefix)
                target = (CACHE_ROOT / str(relative)).resolve()
                if not target.is_relative_to(CACHE_ROOT):
                    raise ValueError("归档内容超出缓存目录")
                destinations.append((member, target))
            for member, target in destinations:
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    manifest_path = CACHE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("找不到缓存；请指定CACHE_DIR或选择demo/tinystories_subset。")
    MANIFEST = json.loads(manifest_path.read_text(encoding="utf-8"))
    if VERIFY_CACHE:
        verify_cache(CACHE_ROOT, MANIFEST)
else:
    kwargs = dict(demo=DATA_SOURCE == "demo", verify=VERIFY_CACHE, hf_token=HF_TOKEN)
    if DATA_SOURCE == "tinystories_subset":
        kwargs.update(max_train_stories=32, max_validation_stories=8)
    MANIFEST = prepare(CFG["data"], **kwargs)

for key in ("max_history", "max_words", "max_target_length"):
    if CFG["model"][key] < MANIFEST["spec"]["data"][key]:
        raise ValueError(f"模型 {key} 小于缓存要求")
if CFG["model"]["vocab_size"] < MANIFEST["bpe_vocab_size"] or CFG["model"]["word_vocab_size"] < MANIFEST["word_vocab_size"]:
    raise ValueError("模型词表无法容纳缓存token ID")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
atomic_json(CFG, RUN_ROOT / "notebook_config.json")
print("数据缓存:", CACHE_ROOT, "demo=", MANIFEST["spec"]["demo"])
display(MANIFEST["stats"])
DATA_READY = True
'''),
    ('markdown', r'''
## 4. 启动训练

此单元直接运行训练，tqdm 以真正参与 CE 的目标 Token 显示进度。
结果写入 `RUN_ROOT`：`metrics.csv/jsonl`、配置、环境、参数统计和检查点。
中断不会保存未完成的更新；恢复时选择已完整落盘的 `.pt`，而不是假设 `latest.pt` 一定最新。
'''),
    ('code', r'''
from train import run as train_run

if not DATA_READY:
    raise RuntimeError("请先完成数据准备单元。")
TRAINING_STATE = None
try:
    TRAINING_STATE = train_run(copy.deepcopy(CFG), str(RESUME_PATH) if RESUME_PATH else None)
except KeyboardInterrupt:
    print("训练已中断。可从本次目录中已完成的检查点恢复。")
finally:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if TRAINING_STATE is not None:
    display(TRAINING_STATE)
print("输出目录:", RUN_ROOT)
'''),
    ('markdown', r'''
## 5. 查看记录和检查点

下面只比较最大的 `tokens_*.pt` 与 `latest.pt` 的实际 Token 计数，避免把较旧的 latest 当成最新。
'''),
    ('code', r'''
import pandas as pd

def latest_complete_checkpoint(folder):
    numbered = sorted(folder.glob("tokens_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    candidates = numbered[-1:] + ([folder / "latest.pt"] if (folder / "latest.pt").is_file() else [])
    records = []
    for path in candidates:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        records.append((int(saved["state"]["target_tokens"]), int(saved["state"]["step"]), path))
        del saved
    return max(records, key=lambda x: (x[0], x[1]))[2] if records else None

LAST_CHECKPOINT = latest_complete_checkpoint(RUN_ROOT)
print("最新完整检查点:", LAST_CHECKPOINT)
if (RUN_ROOT / "metrics.csv").is_file():
    display(pd.read_csv(RUN_ROOT / "metrics.csv").tail(10))
'''),
    ('markdown', r'''
## 6. 可选：本次运行后继续训练

默认跳过。把 `EXTRA_TOKENS` 改为正数并执行本单元，就会从最新完整检查点增加预算，保留优化器、调度、RNG 和采样游标。
另一种方式是在顶部填写 `RESUME_CHECKPOINT`，并把 `BUDGET_TOKENS` 设置为更大的累计总预算。
'''),
    ('code', r'''
EXTRA_TOKENS = 0
if EXTRA_TOKENS < 0:
    raise ValueError("EXTRA_TOKENS 不能为负数")
if EXTRA_TOKENS:
    if LAST_CHECKPOINT is None:
        raise FileNotFoundError("没有可恢复的完整检查点")
    saved = torch.load(LAST_CHECKPOINT, map_location="cpu", weights_only=False)
    next_cfg = copy.deepcopy(saved["config"])
    previous_tokens = int(saved["state"]["target_tokens"])
    next_cfg["train"]["budget_tokens"] = previous_tokens + EXTRA_TOKENS
    next_cfg["train"]["checkpoint_tokens"] = sorted(set(next_cfg["train"]["checkpoint_tokens"] + [previous_tokens + EXTRA_TOKENS]))
    del saved
    TRAINING_STATE = train_run(next_cfg, str(LAST_CHECKPOINT))
    CFG = next_cfg
    LAST_CHECKPOINT = latest_complete_checkpoint(RUN_ROOT)
    atomic_json(CFG, RUN_ROOT / "notebook_config.json")
    assert TRAINING_STATE["target_tokens"] == previous_tokens + EXTRA_TOKENS
    display(TRAINING_STATE)
    print("新检查点:", LAST_CHECKPOINT)
else:
    print("跳过追加训练；需要时修改 EXTRA_TOKENS。")
'''),
    ('markdown', r'''
## 7. 学习曲线

显示真实训练日志；训练吞吐包含采样、传输、累积和完整更新，不是驻 GPU 的性能 Benchmark。
'''),
    ('code', r'''
import matplotlib.pyplot as plt

if (RUN_ROOT / "metrics.csv").is_file():
    metrics = pd.read_csv(RUN_ROOT / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(metrics.target_tokens, metrics.train_nll, "o-", label="Train")
    val = metrics.dropna(subset=["val_nll"])
    if len(val):
        axes[0].plot(val.target_tokens, val.val_nll, "o-", label="Dev subset")
    axes[0].set(xlabel="CE target tokens", ylabel="NLL (nats/token)")
    axes[0].legend()
    axes[1].plot(metrics.target_tokens, metrics.target_tokens_per_second, "o-")
    axes[1].set(xlabel="CE target tokens", ylabel="Training CE tokens/s")
    fig.tight_layout()
    fig.savefig(RUN_ROOT / "training_preview.png", dpi=150)
    plt.show()
else:
    print("尚无训练日志。")
'''),
    ('markdown', r'''
## 8. 开发集评估

默认只评估开发集的前两个 batch，结果明确是子集指标。
把 `DEV_MAX_BATCHES=None` 可评估完整开发集；这里不自动使用 test split。
'''),
    ('code', r'''
RUN_DEV_EVALUATION = True
DEV_MAX_BATCHES = 2
DEV_BATCH_SIZE = 4
if RUN_DEV_EVALUATION and LAST_CHECKPOINT is not None:
    from evaluate import evaluate_checkpoints
    evaluation = evaluate_checkpoints(
        [str(LAST_CHECKPOINT)], str(CACHE_ROOT), str(RUN_ROOT / "dev_eval"),
        split="dev", batch_size=DEV_BATCH_SIZE, device_name=DEVICE,
        max_batches=DEV_MAX_BATCHES, diagnostics=False, top1=False,
    )
    display(pd.DataFrame(evaluation)[["trained_target_tokens", "target_tokens", "nll", "perplexity"]])
    print("评估范围:", "完整 dev" if DEV_MAX_BATCHES is None else f"dev 前 {DEV_MAX_BATCHES} 个 batch")
else:
    print("跳过开发集评估。")
'''),
    ('markdown', r'''
## 保存位置与恢复

数据路径是 `CACHE_ROOT`，输出路径是 `RUN_ROOT`。`CONFIG`相对 `PROJECT_ROOT`，其余相对路径均相对 `STORAGE_ROOT`。
Colab临时磁盘重置会清空运行文件；启用Google Drive后，数据和检查点保存在挂载目录。
恢复时设置 `RESUME_CHECKPOINT="runs/你的目录/tokens_512.pt"`、更大的累计 `BUDGET_TOKENS`；若缓存搬家，再设置 `CACHE_DIR`。

主模型选择 `configs/v3_main_125m.yaml`；CNN同时设置 `CONFIG="configs/ablations/seq_cnn.yaml"` 和 `MODE="seq_cnn4"`。
新功能需已在所选GitHub分支上。正式训练需完整数据，不应把默认32个故事小缓存当作全量TinyStories。
本机若缺py312内核，可在PowerShell执行：

```powershell
& 'E:/anaconda3/envs/py312/python.exe' -m ipykernel install --user --name py312 --display-name "Python (py312 CUDA)"
```
'''),
]

def write_notebook(destination):
    cells = []
    for kind, source in CELLS:
        cell = dict(cell_type=kind, id=uuid.uuid4().hex[:8], metadata={}, source=textwrap.dedent(source).strip()+"\n")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name="Python 3 (select CUDA runtime)", language="python", name="python3"),
                                  language_info=dict(name="python", version="3.12"),
                                  colab=dict(name=Path(destination).name), accelerator="GPU"))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).with_name("Train.ipynb")))
    args = parser.parse_args()
    print(write_notebook(args.output))


if __name__ == "__main__":
    main()
