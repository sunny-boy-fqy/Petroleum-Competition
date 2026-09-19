# 依赖声明与环境约束（v4）

> **可执行的 pip 清单见 [`requirements.txt`](../requirements.txt)**（纯 pip 格式，一行一个包）。
> 本文档说明**为什么**是这些包，以及**不能装**什么。

## 环境基线（平台镜像预装，不可变更）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.11** | 不使用 3.12+ 语法 |
| PyTorch | **2.7.1 + cu128** | 镜像预装；wheel 的 `torch.version.cuda` = **12.8**（编译期 runtime）。`check_env.py` 的 hard 底线是"runtime major == 12"，声明值 12.8 只做 warn；**禁止 `pip install torch`** |
| CUDA | **12.8（驱动能力，非 runtime）** | 指 `nvidia-smi` 头部的 `CUDA Version: 12.8`；不得另装 CUDA / 替换驱动。R4-B1：**不要**拿 `torch.version.cuda` 去比驱动声明值 |
| GPU | 1× A100 80GB (sm_80) | bf16 可用 |
| 系统内存 | 16 GiB | 真正瓶颈 → `num_workers=4` |
| 磁盘 | 30 GB | checkpoint 滚动淘汰 + `assert_disk_headroom` |

## 需要用户 pip 安装的包（**我无法自动安装**）

`requirements.txt` 的 required 段就是这份清单，与
`E0/code/check_env.py::REQUIRED_PY_DEPS` 严格一致（有单测锁定）：

| 包 | 用途 | 缺失后果 |
|---|---|---|
| `numpy` | 口径层唯一硬依赖（解析/评分/契约/Gate/缓存） | 整个管线不可用 |
| `pandas` | 逐折表、训练日志、OOF 汇总 | 分析脚本降级为 csv/JSONL |
| `scipy` | 统计检验（bootstrap/分位数交叉校验） | 自写 bootstrap 仍可用 |
| `scikit-learn` | `roc_auc_score` / `average_precision_score`（E6 原子头 AUC/AP） | 自写 rank-based AUC 兜底 |
| `einops` | E3/E4 序列主干（U-Net/TCN/PatchTF）的 `rearrange` | 需改写成 view/permute |

推荐（可选，缺失自动降级）：`tensorboard` —— 平台任务详情页的"迭代曲线"读
`$TENSORBOARD_LOGDIR`；缺失时只写 JSONL 标量。

**不需要**：`pyarrow`（分片缓存是 `.npz`，没有任何代码 `import pyarrow`）、
`onnx` / `onnxruntime`（CPU 推理主路径是 `torch.load(map_location="cpu")`）。

**版本策略**：只查存在性，**不钉死小版本**（`torch` 的 wheel **不**把 numpy 列为依赖
（`torch 2.7.1` 的 `Requires-Dist` 里没有 numpy，实测从 PyPI 元数据核对），
基础镜像通常自带，但不作为保证 —— 所以 numpy 在 required 清单里由 `setup_deps.sh` 补装；
其余包按 pip 解析出的兼容版本即可）。精确版本由
`versions/locks/cloud_frozen.txt`（云端 `pip freeze` 回填）提供。

## 明确不安装

| 包 | 原因 |
|---|---|
| `torch` / `nvidia-*` / `cuda-*` | 镜像预装且不可替换；`setup_deps.sh` 用 `--dry-run` 预检，命中即中止（exit 3） |
| `torchvision` / `timm` | 图像侧依赖，与测井无关 |
| `flash-attn` / `xformers` / `apex` / `deepspeed` | 需现场编译 CUDA 扩展；注意力走 `F.scaled_dot_product_attention` |
| `matplotlib` / `jupyter` / `wandb` | v4 不做绘图 |

## 降级层

`src/portability.py` 对每个可选库做探测；缺失时走 numpy/标准库兜底（例如 OOF 用
`np.savez_compressed` 代替 parquet，TensorBoard 用 JSONL 代替）。因此**即使某个可选包
装不上，管线也不会崩**；`check_env.py` 会把降级路径记进 `E0_env.json::degraded_paths`。
