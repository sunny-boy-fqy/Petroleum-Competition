# 依赖声明与环境约束（v4）

> 可执行的 pip 清单见 [`requirements.txt`](../requirements.txt)（纯 pip 格式）。
> 本文档说明**为什么**是这些包，以及**不能装**什么。

## 环境基线（平台镜像预装，不可变更）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.11** | 不使用 3.12+ 语法 |
| PyTorch | **2.4.0 + cu124** | 匹配 CUDA 12.6 驱动；**禁止 `pip install torch`** |
| CUDA | **12.6** | 不得另装 CUDA / 替换驱动 |
| GPU | 1× A100 80GB (sm_80) | bf16 可用 |
| 系统内存 | 16 GiB | 真正瓶颈 → `num_workers=4` |
| 磁盘 | 30 GB | checkpoint 滚动淘汰 + `assert_disk_headroom` |

## 额外轻量包（允许 pip，`--no-cache-dir`）

`pandas 2.2.3` / `pyarrow 17.0.0` / `scipy 1.13.1` / `scikit-learn 1.5.2` /
`einops 0.8.0` / `onnx 1.16.2` / `onnxruntime 1.18.1`

## 明确不安装

| 包 | 原因 |
|---|---|
| `torchvision` / `timm` | 图像侧依赖，与测井无关 |
| `flash-attn` / `xformers` / `apex` / `deepspeed` | 需现场编译 CUDA 扩展；注意力走 `F.scaled_dot_product_attention` |
| `tensorboard` | 平台已集成（写 `TENSORBOARD_LOGDIR`）；训练日志用 CSV/JSONL |
| `matplotlib` / `jupyter` / `wandb` | v4 不做绘图 |

## 降级层

`src/portability.py` 对每个可选库做探测；缺失时走 numpy/标准库兜底（例如 OOF 用
`np.savez_compressed` 代替 parquet）。因此**即使某个可选包装不上，管线也不会崩**。
