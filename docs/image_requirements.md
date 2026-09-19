# 训练镜像需求（平台「镜像」功能）

> 依据平台文档：[镜像](http://discovery-staging.intern-ai.org.cn/docs/workbench/images)。
> 平台镜像用**快捷安装**（apt / pip）或 **Dockerfile 编辑**构建；
> **使用场景必须选「训练任务」**（选错后训练任务里看不到该镜像）；
> 平台**最多 5 个镜像**，因此 v4 只需要 **1 个**训练镜像。

---

## 一、环境基线（**不可变更**）

| 组件 | 要求 | 说明 |
|---|---|---|
| CUDA | **12.6** | 平台镜像自带；**不得另装 CUDA / 替换驱动** |
| PyTorch | **2.4.0 + cu124** | 平台预装；**不得 `pip install torch` 改版本** |
| Python | **3.11** | 平台预装；不得换 3.12+ |
| GPU | 1× **Nvidia A100 80 GB**（sm_80） | 资源规格选择；bf16 可用 |
| 系统内存 | 16 GiB | **这是真正的瓶颈**，不是显存 → `DataLoader(num_workers=4)` |
| 磁盘 | 30 GB | checkpoint 滚动淘汰 + `assert_disk_headroom(8.0)` |

代码侧的双保险：`v4/E0/code/check_env.py` 会硬断言上述版本/设备，不通过即非零退出；
`v4/src/portability.py` 对每个可选库做"探测 + 降级"，缺失也不会崩。

---

## 二、方案 A（推荐）：基于平台官方 PyTorch 镜像 + 快捷安装

### 2.1 选择基础镜像

新建镜像 → 使用场景**训练任务** → 资源配置选 **Nvidia GPU（A100）** → 基础镜像下拉里选
**PyTorch 2.4.0 / CUDA 12.6 / Python 3.11** 对应的那一项。

> 下拉项名称由平台给（例如 `pytorch:2.4.0-cuda12.6-cudnn9-py311` 这类写法），
> **请以页面上实际显示的为准**；只要三个版本号对得上即可。
> 若列表里没有恰好匹配的版本，改用【方案 B：Dockerfile】以显式锁定。

### 2.2 快捷安装内容

【快捷安装】→ 添加 **pip** 依赖，每行一个（**只填包名与版本，不要写 `pip install`**）：

```
pandas==2.2.3
pyarrow==17.0.0
scipy==1.13.1
scikit-learn==1.5.2
einops==0.8.0
onnx==1.16.2
onnxruntime==1.18.1
```

**不要安装**（体积 / 兼容 / 无必要）：

| 包 | 原因 |
|---|---|
| `torch` / `torchvision` / `timm` | 会替换平台预装版本，或引入数 GB 无关依赖 |
| `nvidia-*` / `cuda-*` | 重复下载 CUDA 运行时，30 GB 磁盘装不下 |
| `flash-attn` / `xformers` / `apex` / `deepspeed` | 需现场编译 CUDA 扩展，构建易失败；注意力统一用 PyTorch 2.4 原生 `F.scaled_dot_product_attention` |
| `tensorboard` | 平台已集成（写 `TENSORBOARD_LOGDIR` 即可）；训练日志用 CSV/JSONL |
| `matplotlib` / `jupyter` / `wandb` | v4 不做绘图，指标一律落 JSON/CSV |

### 2.3 可选：apt 依赖

v4 **不需要**任何 apt 包（解析器只用标准库 + numpy）。

---

## 三、方案 B：Dockerfile 编辑（需要显式锁版本时）

在【Dockerfile编辑】中，基础镜像的 `FROM` 由平台自动填入，**保留它**，在后面追加：

```dockerfile
# ==== v4 训练镜像追加层 ====
# 注意：不要 apt install / pip install 任何 CUDA 或 torch 相关包
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir \
      "pandas==2.2.3" \
      "pyarrow==17.0.0" \
      "scipy==1.13.1" \
      "scikit-learn==1.5.2" \
      "einops==0.8.0" \
      "onnx==1.16.2" \
      "onnxruntime==1.18.1" \
 && python -m pip cache purge

# 自检：版本不对就让镜像构建失败，而不是等到训练时才炸
RUN python - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split("+")[0] == "2.4.0", torch.__version__
print("v4 image OK:", sys.version.split()[0], torch.__version__, torch.version.cuda)
PY
```

---

## 四、镜像构建后自检（在任意训练任务里跑）

```bash
bash /code/workspace/v4/run_train.sh --mode env
```

期望输出（`check_env.py` 的 hard 检查全过）：

```
[OK  ] python_version          python 3.11.x (expected 3.11)
[OK  ] torch_version           torch 2.4.0+cu124 (expected 2.4.0)
[OK  ] cuda_runtime_version    torch.version.cuda=12.4 (expected 12.4 = torch 2.4.0+cu124 runtime; 平台驱动能力见 cuda_driver_version)
[WARN] cuda_driver_version     nvidia-smi CUDA Version=12.6 (平台声明 12.6；驱动能力由平台保证，advisory 不阻塞)
[OK  ] cuda_available          torch.cuda.is_available()=True
[OK  ] gpu_is_a100             NVIDIA A100-SXM4-80GB sm_80 79.3 GiB
[OK  ] bf16_supported          torch.cuda.is_bf16_supported()=True
[OK  ] disk_headroom           ... free=XX GiB (require >= 8 GiB)
hard failures: 0
```

> **CUDA 语义（R4-B1，务必分清）**：平台镜像是 `torch==2.4.0+cu124`，该 wheel 的
> `torch.version.cuda` 恒为 **12.4**；镜像文档里的 **CUDA 12.6** 指的是**驱动能力**
> （`nvidia-smi` 头部的 `CUDA Version: 12.6`），两者不是同一个数。`check_env.py` 现在
> 用 `cuda_runtime_version` 硬校验 12.4、用 `cuda_driver_version` 以 **warn** 提示 12.6；
> 旧版拿 `torch.version.cuda` 硬比 12.6 会让云端 `--mode env` 必然 `exit 11`，
> 使 `E0_cloud_gate` 永远点不亮。`E0_env.json::expected` 也分别给出
> `cuda_runtime` 与 `cuda_driver_min` 两个键。

结果写入 `/data/v4/reports/E0_env.json`；磁盘分布写入 `E0_disk_budget.json`。

---

## 五、依赖版本快照（提交与复现用）

镜像构建后，在任务里执行并把输出纳入版本控制：

```bash
python -m pip freeze > /data/v4/reports/cloud_frozen.txt
# 与 v4/versions/locks/cloud.txt 比对；不一致时更新 lock 文件并提交
```

| lock 文件 | 用途 | 内容 |
|---|---|---|
| `v4/versions/locks/cloud.txt` | 云端**训练**环境 | torch 2.4.0+cu124（镜像提供）+ 上述 pip 包 |
| `v4/versions/locks/submit.txt` | **提交推理**环境 | torch 2.4.0 + numpy + pandas（+ 可选 onnx/onnxruntime） |

---

## 六、镜像相关 FAQ（来自平台文档）

1. **为什么切换资源配置后基础镜像列表变了？** 平台按 CPU/GPU/NPU 过滤兼容镜像 → 先选 GPU 资源，再选带 CUDA 的 PyTorch 镜像。
2. **构建失败怎么查？** 【查看日志】看末尾错误，重点核对包名/版本是否存在、是否与基础镜像冲突、是否误装 CUDA 包。
3. **镜像在训练任务里选不到？** 检查使用场景是否选了「训练任务」，以及资源是否与该镜像的基础环境兼容。
4. **镜像数量上限？** 5 个；用不到就删（删除不可恢复，但**不影响正在运行的任务**）。
5. **训练任务与开发机镜像通用吗？** **不通用**；v4 只需建 1 个「训练任务」镜像。
