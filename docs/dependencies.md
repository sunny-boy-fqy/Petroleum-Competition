# 依赖声明与环境约束（v4）

> **可执行的 pip 清单见 [`requirements.txt`](../requirements.txt)**（纯 pip 格式，一行一个包）。
> 本文档说明**为什么**是这些包，以及**不能装**什么。

## 环境基线（平台镜像预装，不可变更）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.11** | 不使用 3.12+ 语法 |
| 架构 | **aarch64 / arm64** | 目标机是 arm64；镜像里必须是 aarch64 轮子（本机 x86_64 只用 `--allow-non-target-device` 跑口径层） |
| PyTorch | **2.8.0** | 镜像预装；`check_env.py` 的 hard 底线是 **major.minor == 2.8**，补丁号/build 串漂移只 warn；**禁止 `pip install torch`** |
| torch_npu | **2.8.0** | 镜像预装，**必须与 torch 同小版本**；由它注册 `torch.npu`，缺它 NPU 路径不可用；**禁止 `pip install torch_npu`** |
| CANN | **8.3rc2** | Ascend 运行时/工具包；hard 底线 **major.minor == 8.3**，rc/补丁漂移只 warn（R4-B1：**不要**把具体 rc 钉成 hard 断言） |
| NPU | 1× **Ascend 910B（64 GB HBM）** | bf16 可用（`check_env.py` **实测**一次 bf16 matmul，不假定存在 `is_bf16_supported`） |
| 系统内存 | 16 GiB | 真正瓶颈 → `num_workers=4` |
| 磁盘 | 64 GiB | checkpoint 滚动淘汰 + `assert_disk_headroom` |

> 硬件画像的**唯一事实源**是 [`src/hardware.py`](../src/hardware.py)（`PLATFORM` / `describe()`）；
> `check_env.py`、本文档、`docs/image_requirements.md` 与生成的计划都由它派生，
> 并由 `tests/test_hardware.py` 锁定不得各自漂移。

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

**版本策略**：只查存在性，**不钉死小版本**。`torch` 的 wheel **不**把 numpy 列为依赖
（`torch 2.8.0` 的 `Requires-Dist` 里没有 numpy —— 已从 PyPI 元数据核对），基础镜像通常
自带但不保证，所以 numpy 也在 required 清单里由 `setup_deps.sh` 补装；其余包按 pip 解析出的
兼容版本即可。

## 版本到底在哪里被钉死（两阶段流程）

| 阶段 | 动作 | 谁执行 |
|---|---|---|
| ① 首次装 | `bash v4/E0/code/setup_deps.sh`（不带参数）→ 只给**包名**，让 pip 按当前镜像解析 | 用户（云端 `--mode env` 自动调用） |
| ② 记录事实 | 脚本第 3 步自动 `pip freeze > $V4_REPORTS_DIR/cloud_frozen.txt`，并回拷 `versions/locks/cloud_frozen.txt` | 脚本 |
| ③ 需要确定性时 | `bash v4/E0/code/setup_deps.sh --from-frozen` → 按 ② 的**精确版本**安装（重建镜像 / 复现提交） | 用户 |

**为什么首次不钉**：`python 3.11 + 平台镜像`组合下 pip 会解析出哪个小版本，只有实机才知道；
我（agent）无法安装、无法探测，猜错会让镜像构建失败或引入 ABI 冲突 —— 这与四审"把 CUDA
runtime 小版本钉死导致 Gate 永久 blocked"、以及把 CANN 的 rc 后缀钉死是同一类错误。

**`--from-frozen` 的安全边界**：`E0/code/frozen_pins.py` 用**白名单**（required 5 个 +
可选 `tensorboard`）从 freeze 里挑包，因此结果**在构造上不可能**包含 `torch` / `torch_npu` /
`npu*` / `ascend*` / `cann*` / `nvidia-*` / `cuda-*` / `triton`
（禁用前缀的单一事实源 = `src/hardware.py::FORBIDDEN_INSTALL_PREFIXES`）；
`pip` / `setuptools` / `wheel` 与全部传递依赖也会被忽略。
可用 `V4_FROZEN_LOCK=<path>` 指向另一份 freeze 输出。

**明确不钉的东西**：不钉 `numpy` 的精确版本（除非 frozen 证明镜像里就是那个版本）。
`numpy` 1.x / 2.x 我们的代码都兼容（已核查无 `np.float_` / `np.NaN` / `np.trapz` 等被移除别名），
而把一个旧 numpy 强装进已装 numpy 2.x 的镜像，会让 scipy/scikit-learn 的 wheel 与 numpy ABI 打架。

## 明确不安装

| 包 | 原因 |
|---|---|
| `torch` / `torch_npu` / `npu*` / `ascend*` / `cann*` / `nvidia-*` / `cuda-*` / `triton` | 镜像预装且不可替换（Ascend 侧 `torch.npu` 依赖 torch 与 torch_npu 的**同小版本**匹配，乱装必崩）；`setup_deps.sh` 用 `--dry-run` 预检，命中即中止（exit 3） |
| `torchvision` / `timm` | 图像侧依赖，与测井无关 |
| `flash-attn` / `xformers` / `apex` / `deepspeed` | 需现场编译 NPU/CUDA 扩展；注意力走 `F.scaled_dot_product_attention` |
| `matplotlib` / `jupyter` / `wandb` | v4 不做绘图 |

## 降级层

`src/portability.py` 对每个可选库做探测；缺失时走 numpy/标准库兜底（例如 OOF 用
`np.savez_compressed` 代替 parquet，TensorBoard 用 JSONL 代替）。因此**即使某个可选包
装不上，管线也不会崩**；`check_env.py` 会把降级路径记进 `E0_env.json::degraded_paths`。
