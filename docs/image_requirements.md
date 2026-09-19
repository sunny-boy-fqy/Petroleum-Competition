# 训练镜像需求（平台「镜像」功能）

> 依据平台文档：[镜像](http://discovery-staging.intern-ai.org.cn/docs/workbench/images)。
> 平台镜像用**快捷安装**（apt / pip）或 **Dockerfile 编辑**构建；
> **使用场景必须选「训练任务」**（选错后训练任务里看不到该镜像）；
> 平台**最多 5 个镜像**，因此 v4 只需要 **1 个**训练镜像。

---

## 一、环境基线（**不可变更**）

| 组件 | 要求 | 说明 |
|---|---|---|
| CUDA | **12.8** | 平台镜像自带；**不得另装 CUDA / 替换驱动** |
| PyTorch | **2.7.1 + cu128** | 平台预装；**不得 `pip install torch` 改版本** |
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
**PyTorch 2.7.1 / CUDA 12.8 / Python 3.11** 对应的那一项。

> 下拉项名称由平台给（例如 `pytorch:2.7.1-cuda12.8-cudnn9-py311` 这类写法），
> **请以页面上实际显示的为准**；只要三个版本号对得上即可。
> 若列表里没有恰好匹配的版本，改用【方案 B：Dockerfile】以显式锁定。

### 2.2 快捷安装内容

【快捷安装】→ 添加 **pip** 依赖，每行一个（**只填包名，不要写 `pip install`、不要写版本**）：

```
numpy
pandas
scipy
scikit-learn
einops
tensorboard
```

> **版本不钉死（R5-M1）**：基础镜像通常自带 numpy，但 `torch` 的 wheel **并不依赖 numpy**
> （`torch 2.7.1` 的 `Requires-Dist` 无 numpy），所以 numpy 仍列在 required 里由 pip 补装；
> 其余包按构建时 pip 解析出的兼容版本即可。精确版本由镜像构建后回填的
> `versions/locks/cloud_frozen.txt`（`pip freeze`）提供 —— 把具体小版本写进镜像
> 会在基础镜像升级时构建失败。`tensorboard` 只是为了让平台任务详情页能看到
> "迭代曲线"；不装也能跑（自动降级为 JSONL 标量）。

**不要安装**（体积 / 兼容 / 无必要）：

| 包 | 原因 |
|---|---|
| `torch` / `torchvision` / `timm` | 会替换平台预装版本，或引入数 GB 无关依赖 |
| `nvidia-*` / `cuda-*` | 重复下载 CUDA 运行时，30 GB 磁盘装不下 |
| `flash-attn` / `xformers` / `apex` / `deepspeed` | 需现场编译 CUDA 扩展，构建易失败；注意力统一用 PyTorch 2.7 原生 `F.scaled_dot_product_attention` |
| `pyarrow` | 分片缓存是 `.npz`（numpy），没有任何代码 `import pyarrow` |
| `onnx` / `onnxruntime` | CPU 推理主路径是 `torch.load(map_location="cpu")`；缺失自动降级 |
| `matplotlib` / `jupyter` / `wandb` | v4 不做绘图，指标一律落 JSON/CSV |

> **我（agent）无法自动安装任何包**：上面这份清单需要用户在自己的环境里执行 `pip install`。
> 清单与 `E0/code/check_env.py::REQUIRED_PY_DEPS` 严格一致，并由
> `tests/test_platform_scripts.py::test_pip_install_list_matches_declared_deps` 锁定。

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

# required 清单与 requirements.txt 的**有效行**逐字一致（有单测锁定）。
# 刻意**不写版本**：基础镜像升级时 pip 解析出的兼容版本即可；精确版本由
# 构建后 `pip freeze` 回填 versions/locks/cloud_frozen.txt（见第五节）。
RUN python -m pip install --no-cache-dir \
      numpy pandas scipy scikit-learn einops \
 && python -m pip cache purge

# 可选：平台"迭代曲线"观测；缺失时训练侧自动降级为 JSONL 标量，不阻塞
RUN python -m pip install --no-cache-dir tensorboard \
 && python -m pip cache purge

# 自检：版本不对就让镜像构建失败，而不是等到训练时才炸
RUN python - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
print("v4 image OK:", sys.version.split()[0], torch.__version__, torch.version.cuda)
PY
```

> **不要**装 `onnx` / `onnxruntime` / `pyarrow`：分片缓存是 `.npz`（numpy），CPU 推理主路径是
> `torch.load(map_location="cpu")`；二者只在 `src/portability.py` 里作为**可选兜底**被探测，
> 缺失时走文档化降级路径。把它们装进镜像既浪费磁盘，也会和 `requirements.txt` /
> `E0/code/setup_deps.sh` 的清单打架。

---

## 四、镜像构建后自检（在任意训练任务里跑）

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env
```

期望输出（`check_env.py` 的 hard 检查全过）：

```
[OK  ] python_version          python 3.11.x (expected 3.11)
[OK  ] torch_version           torch 2.7.1+cu128 (expected 2.7.1)
[OK  ] cuda_runtime_version    torch.version.cuda=12.8 (hard 要求 runtime major==12；声明值 12.8，可接受 ['12.8', '12.6'])
[OK  ] cuda_runtime_declared   torch.version.cuda=12.8 vs 声明 12.8（warn 级；可接受 ['12.8', '12.6'] 内的任意值，小版本漂移不阻塞）
[OK  ] cuda_driver_version     nvidia-smi CUDA Version=12.8 (平台声明 12.8；warn/advisory 级，驱动能力由平台保证，不阻塞)
[OK  ] cuda_available          torch.cuda.is_available()=True
[OK  ] gpu_is_a100             NVIDIA A100-SXM4-80GB sm_80 79.3 GiB
[OK  ] bf16_supported          torch.cuda.is_bf16_supported()=True
[OK  ] disk_headroom           ... free=XX GiB (require >= 8 GiB)
hard failures: 0
```

> **状态列怎么读（R5-L1）**：`check_env.py` 的规则是
> `mark = "OK  " if ok else ("FAIL" if level == "hard" else "WARN")` ——
> 即 **`[WARN]` 只在 warn 级检查*失败*时出现**。`cuda_runtime_declared` 与
> `cuda_driver_version` 是 warn 级，值对得上时打印 **`[OK  ]`**，只有漂移时才打印
> `[WARN]` 且**不阻塞**（`hard failures: 0` 仍然是唯一判据）。

> **CUDA 语义（R4-B1 + R5-B1，务必分清）**：平台镜像是 `torch==2.7.1+cu128`，该 wheel 的
> `torch.version.cuda` 是 **12.8**；镜像文档里的 **CUDA 12.8** 也指**驱动能力**
> （`nvidia-smi` 头部的 `CUDA Version: 12.8`），两者名字相同但来源不同，必须分别读。
> `check_env.py` 现在是三层口径：**hard** = `torch.version.cuda` 存在且 major == 12；
> **warn** = 是否等于声明值 12.8（`cuda_runtime_declared`）；**advisory** =
> `nvidia-smi` 的 `CUDA Version >= 12.8`（`cuda_driver_version`）。
> 四审的历史教训：硬断言某个具体 runtime（12.4 / 12.6）会让云端 `--mode env` 必然
> `exit 11`，`E0_cloud_gate` 永远点不亮；因此**不得**再把 wheel 小版本写成 hard 断言。
> `E0_env.json::expected` 给出 `cuda_runtime`、`cuda_runtime_accepted`、
> `cuda_runtime_hard_major`、`cuda_driver_min` 四个键。

结果写入 `/data/v4/reports/E0_env.json`；磁盘分布写入 `E0_disk_budget.json`。

---

## 五、依赖版本快照（提交与复现用）

**两阶段版本策略**（详见 `docs/dependencies.md`）：

1. **首次装**（`run_train.sh --mode env` 会自动调用 `setup_deps.sh`）：只给**包名**，
   由 pip 按当前镜像解析兼容版本 —— 我不会猜小版本号，猜错会让镜像构建失败；
2. **记录事实**：脚本装完立即

   ```bash
   python -m pip freeze > /data/v4/reports/cloud_frozen.txt
   # 同时回拷到 v4/versions/locks/cloud_frozen.txt（进 git，供提交复现）
   ```

3. **需要精确复现时**（重建镜像 / 打包提交）：按实测版本安装

   ```bash
   bash v4/E0/code/setup_deps.sh --from-frozen
   ```

   `E0/code/frozen_pins.py` 用白名单（required 5 个 + 可选 `tensorboard`）从 freeze 里挑包，
   结果**在构造上不可能**含 `torch` / `nvidia-*` / `cuda-*` / `triton`；
   传递依赖与 `pip`/`setuptools` 一律忽略。

| lock 文件 | 用途 | 内容 |
|---|---|---|
| `v4/versions/locks/cloud.txt` | 云端**训练**环境 | torch 2.7.1+cu128（镜像提供）+ 上述 pip 包（**不钉版本**） |
| `v4/versions/locks/submit.txt` | **提交推理**环境 | torch 2.7.1+cu128（镜像提供）+ numpy（推理侧不需要训练专属包） |
| `v4/versions/locks/cloud_frozen.txt` | **精确复现**（`--from-frozen`） | 云端 `pip freeze` 回填的真实版本 |

---

## 六、镜像相关 FAQ（来自平台文档）

1. **为什么切换资源配置后基础镜像列表变了？** 平台按 CPU/GPU/NPU 过滤兼容镜像 → 先选 GPU 资源，再选带 CUDA 的 PyTorch 镜像。
2. **构建失败怎么查？** 【查看日志】看末尾错误，重点核对包名/版本是否存在、是否与基础镜像冲突、是否误装 CUDA 包。
3. **镜像在训练任务里选不到？** 检查使用场景是否选了「训练任务」，以及资源是否与该镜像的基础环境兼容。
4. **镜像数量上限？** 5 个；用不到就删（删除不可恢复，但**不影响正在运行的任务**）。
5. **训练任务与开发机镜像通用吗？** **不通用**；v4 只需建 1 个「训练任务」镜像。
