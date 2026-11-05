# legacy/ — 历史版本的验证、基准与研究脚本

这里集中存放已完成并归档的各阶段实验脚本。它们仍然可以运行，并被
`tests/` 以及当前的 V3.5 脚本（`benchmark_cky_v35.py` 等）作为库导入，
但不再是日常开发入口。当前入口保留在仓库根目录：`train.py`、`evaluate.py`、
`profile_performance.py`、`*_cky_v35.py`、`local_cnn_ablation.py`、`build_train_notebook.py`。

| 目录 | 阶段 | 内容 | 报告 |
| --- | --- | --- | --- |
| `cky_v2/` | CKY V2 score-first | `validate_cky_v2.py`、`benchmark_cky_v2.py` | `reports/cky_v2/` |
| `cky_v25/` | CKY V2.5 lazy_semantic | `validate_cky_v25.py`、`benchmark_cky_v25.py` | `reports/cky_v25/` |
| `precision_v25/` | V2.5-P 精度研究 | `run_precision_study.py`（总入口）、`validate_precision_*.py`、`benchmark_precision_v25.py`、`precision_study.py`、`precision_supplement.py`、`precision_trace.py`、`train_precision_stability.py`、`summarize_precision_study.py`、`PrecisionStudy_Colab.ipynb` | `reports/cky_v25_precision/` |
| `reader_v25/` | V2.5-R Reader 精度隔离 | `reader_precision_study.py`、`validate_reader_precision.py`、`benchmark_reader_precision.py`、`audit_reader_search.py`、`audit_reader_profiler.py`、`summarize_reader_precision.py` | `reports/cky_v25_reader/` |
| `cky_v3/` | CKY V3 width-batched tensorization | `v3_study.py`、`validate_cky_v3.py`、`train_cky_v3_validation.py`、`benchmark_cky_v3.py`、`profile_cky_v3_widths.py`、`summarize_cky_v3.py` | `reports/cky_v3/` |

## 运行方式

所有脚本都假设**当前工作目录是仓库根目录**（`configs/`、`cache/`、`reports/` 等相对路径由此解析）。
每个脚本开头有一段把仓库根目录加入 `sys.path` 的引导代码，因此下面两种写法都可以：

```powershell
& 'E:/anaconda3/envs/py312/python.exe' legacy/cky_v3/validate_cky_v3.py
& 'E:/anaconda3/envs/py312/python.exe' -m legacy.cky_v3.validate_cky_v3
```

作为库导入时使用完整包路径，例如 `from legacy.cky_v3.v3_study import forest_pair`。

`precision_study_cache.zip` 和 `requirements-precision.txt` 仍留在仓库根目录，
因为 `build_train_notebook.py` 生成的 Notebook 会在根目录查找该缓存包。
