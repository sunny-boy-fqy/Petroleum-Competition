# V4 六审报告

- **审查对象**：`fb2e5b9`（五审 R5-B1/H1/H2/M1/M2/L1..L3 返工）
- **基线**：`1dce27e` -> `fb2e5b9`，工作树干净
- **审查方式**：不信声明，先跑当前提交的测试、工具、干净 archive、云端数据部署路径、文档静态一致性
- **结论**：**R5 七项修复全部闭环，云端 `env -> data -> e0` 已具备开跑条件；本审另发现 2 个中等逻辑不一致 + 3 个低危项，均不是当前 E0 云端阻塞项，但 R6-M1 应在 E10 权重导出实现前修掉。**

---

## 一、独立复核通过项

### 1.1 测试与证据

在仓库内用两个解释器分别运行 `tests/run_all.py`：

```text
.venv-torch/bin/python tests/run_all.py
运行 214 项，失败 0，错误 0，跳过 0

../v2/.venv/bin/python tests/run_all.py
运行 214 项，失败 0，错误 0，跳过 37
```

干净 `git archive`（不含被 gitignore 的 `dist/*`、`../data/`）：

```text
无 torch：209 项，失败 0，跳过 49
有 torch：209 项，失败 0，跳过 12
```

新增 19 项回归确实覆盖了 R5 的四条要求。其中以下关键项已独立复现：

- 空数据 `check_data_leak.py`：`RESULT: FAIL`，`rc=1`。
- 正常数据：`检查井数: 80+10=90，违规: 0，RESULT: OK`。
- `bootstrap_data.sh --tarball <云盘路径>`：能解压、校验 80/10 与 730,268/95,948。
- 无 tarball：`exit 4`，并列出搜索过的全部位置。
- 文档/PLAN/requirements/lock/setup_deps 的 required 集合一致。

### 1.2 R5-B1 云端部署路径

我在本机把 repo 的 `dist/v4_data.tar.gz` 移出搜索范围，只把 tarball 放到
“云盘根” `$DATA_ROOT/v4_data.tar.gz`，然后执行 **无参数** 的正式入口：

```bash
V4_DATA_ROOT=<tmp-data-root> bash run_train.sh --mode data
```

部署阶段实际输出：

```text
tarball    : <tmp-data-root>/v4_data.tar.gz
manifest   : <tmp-data-root>/v4_data_manifest.json
tarball sha256 OK
train wells=80 rows=730268  (expect 80 / 730268)
test  wells=10 rows=95948   (expect 10 / 95948)
RESULT: OK
```

随后 `check_env --profile full` 因本机开发环境无 torch / Python 3.12 / 空闲盘不足而返回 13；
数据相关项 `data_train_80`、`data_test_10` 均为 `[OK]`。这验证了 R5-B1 的核心路径：
**云端 repo 内没有 dist/ 时，把 tarball 放到 `/data` 根，无参数 `--mode data` 能找到它。**

### 1.3 R5-H1 / H2 验证

```text
# 空目录
检查井数: 0+0=0（期望 train=80 / test=10），输入列数要求: 13
违规: 2
  - coverage[train]: ...
  - coverage[test]: ...
RESULT: FAIL
rc=1
```

`bootstrap_data.sh` 的硬校验：

- 井数 80/10 与行数 730,268/95,948 均来自 `src.constants`；
- 没有 manifest 时，行数校验仍然无条件执行；
- manifest 存在时，先校 tarball sha256，再校逐项计数；
- 无 manifest、tarball 截断或行数不符时无法通过。

这一项与声明一致。

### 1.4 R5-M1 / M2

- `PLAN.md`、`docs/image_requirements.md`、`requirements.txt`、`versions/locks/cloud.txt`、
  `E0/code/setup_deps.sh` 的 required 集合已统一为：
  `numpy / pandas / scipy / scikit-learn / einops`。
- `tensorboard` 已明确为可选推荐，缺失时降级 JSONL。
- `pyarrow/onnx/onnxruntime` 已从 required/Dockerfile 安装命令移出。
- `setup_deps.sh` lock 过滤只排除 `torch`，`numpy` 回到安装清单。
- `setup_deps.sh --dry-run` 确认不再写 `E0_env.json` / `E0_disk_budget.json`。

### 1.5 R5-L1..L3

- `docs/image_requirements.md` 期望输出已改为 `[OK  ] cuda_runtime_declared`、
  `[OK  ] cuda_driver_version`，并写明 `[WARN]` 只在 warn 级失败时出现。
- `src/constants.py` 注释已改为“14 个输入字段 = DEPTH + 13 条曲线”。
- `tools/bootstrap_data.sh` tar 内路径注释已改为 `v4/data/...`。

---

## 二、本审新发现

### R6-M1（中）：ONNX 口径仍自相矛盾

**位置**

- `PLAN.md:262`：`torch.onnx.export`（torch 2.7 内置）
- `PLAN.md:760`：硬要求“**优先导出 ONNX** 作为兜底路径”
- `E10/P0/PLAN.md:15,31,38,50,70`
- 但 `PLAN.md:163,217,225` 与 `requirements.txt` 又明确写
  `onnx` / `onnxruntime` **不需要、不安装**。

**硬证据**

在带 torch 2.4 但无 `onnx`/`onnxscript` 的 `.venv-torch` 中，对一个最小模型执行
`torch.onnx.export`：

```text
OnnxExporterError: Module onnx is not installed!
```

torch 2.7.1 的 ONNX 导出同样至少需要额外安装 `onnx`；若走 dynamo exporter，还需要
`onnxscript`。因此“torch 内置 ONNX、优先导出 ONNX”在“不安装 onnx”的依赖政策下不成立。

**建议**

二选一，不要保留现状：

1. 把 `onnx`（以及 torch 2.7 需要的 `onnxscript`）列为**可选安装**，并在计划里说明磁盘代价；
2. 更符合当前“主路径 torch CPU 推理”的方案：把措辞改成
   “若用户额外安装了 onnx，则 best-effort 尝试导出 ONNX；否则不导出，实际兜底是 `.pt`/`.npz`”。

无论选哪个，都必须同步改 `PLAN.md:262,760` 与 `E10/P0/PLAN.md`。

---

### R6-M2（中）：单一事实源未完全贯彻

`src/constants.py` 的模块注释明确说：

> 任何模块需要数据列名、哨兵值、占位常量、评分权重或折路径时，都必须从这里取，
> 不得在别处硬编码字面量。

R5-H2 新增了：

```python
EXPECTED_N_TRAIN_WELLS = 80
EXPECTED_N_TRAIN_ROWS = 730_268
```

但本审 grep 发现仍有两处硬编码：

- `E0/code/run_all.py:578`

```python
"row_counts_match": card_train["n_rows"] == 730_268 and ...
```

- `E0/code/check_env.py:376-379`

```python
rep.add("data_train_80", n_train == 80, ...)
rep.add("data_test_10", n_test == 10, ...)
```

这两个文件正好是 E0 Gate 与云端环境 Gate 的判据来源，继续保留字面量会让
“常量一改、Gate 口径不同步”的风险重新出现。

**建议**

- `run_all.py` 改用 `C.EXPECTED_N_TRAIN_ROWS` / `C.EXPECTED_N_TEST_ROWS`；
- `check_env.py` 可在显式 `sys.path.insert(0, str(root))` 后 import `src.constants`
  （该模块只依赖标准库），改用 `C.EXPECTED_N_TRAIN_WELLS` / `C.EXPECTED_N_TEST_WELLS`；
- 把静态回归从只扫描 `bootstrap_data.sh` 扩展到扫描 `E0/code/*.py` 中的
  `80` / `10` / `730_268` / `95_948` 相关硬编码。

---

### R6-L1（低）：显式传入不存在的 manifest 会被静默忽略

当前 `bootstrap_data.sh` 只区分“manifest 是否非空”，不区分“显式指定”与“自动搜索”：

```bash
V4_DATA_ROOT=<tmp> bash tools/bootstrap_data.sh \
  --tarball <tmp>/v4_data.tar.gz \
  --manifest <tmp>/no_such_manifest.json
```

实际结果：

```text
manifest   : (未找到，跳过 sha256 校验；行数/井数仍会硬校验)
train wells=80 rows=730268  (expect 80 / 730268)
RESULT: OK
rc=0
```

用户显式传入 manifest 时，路径写错不应该等于“没有 manifest”。建议：

- `--manifest` / `V4_DATA_MANIFEST` 显式给出但文件不存在 -> `exit 4`；
- manifest 存在但缺少 `tarball.sha256` -> `exit 3`，而不是打印 `tarball sha256 OK`；
- 自动搜索不到时才允许走“无 manifest”分支。

---

### R6-L2（低）：`--verify-only` 仍会写目录，`--dest` 语义不完整

- `tools/bootstrap_data.sh` 在分支之前执行 `mkdir -p "$DEST"`，所以
  `--verify-only` 声称“只校验，不写入”，实测仍会创建 `$DATA_ROOT/v4/data`。
- `--dest` 在 tarball 模式下没有被真正使用：解压始终 `-C "$DATA_ROOT"`，
  校验却读 `$DEST`。如果 `$DATA_ROOT` 不存在，`tar` 直接失败；如果存在，
  数据落到 `$DATA_ROOT/v4/data`，而校验读自定义 `$DEST`，语义不一致。

建议把 `mkdir -p "$DEST"` 移到确实要写数据的 branch，并明确：
要么让 `--dest` 同时控制解压根目录，要么从 tarball 模式中拒绝 `--dest`。

---

### R6-L3（低）：`docs/platform_setup.md` 仍留了一个旧示例路径

`docs/platform_setup.md:67` 写：

```text
把它上传到平台云盘（即可在 /data 看到）即可，例如放到 /v4_data/v4_data.tar.gz
```

但同一文档后面和代码定位协议都要求：

```text
$DATA_ROOT/v4_data.tar.gz      # 推荐，$DATA_ROOT=/data
```

若用户照第 67 行的 `/v4_data/v4_data.tar.gz` 上传，无参数 `--mode data` 搜索不到，
必须显式 `--tarball`。建议改为 `/data/v4_data.tar.gz`，或明确写出 `--tarball` 命令。

---

## 三、审查结论与建议顺序

### 结论

**R5-B1 / R5-H1 / R5-H2 / R5-M1 / R5-M2 / R5-L1..L3 全部通过独立复核。**
云端 E0 的阻塞点已解除，可以按：

```bash
bash /code/workspace/v4/run_train.sh --mode env
bash /code/workspace/v4/run_train.sh --mode data
bash /code/workspace/v4/run_train.sh --mode e0
```

开跑并收集云端 Gate 证据。

本审新增问题均不是当前 `env -> data -> e0` 的阻塞项，其中：

- **R6-M1** 必须在 E10 训练/导出脚本实现前修，否则计划里的 ONNX 兜底无法执行；
- **R6-M2** 建议本批一起修，避免 Gate 口径再次漂移；
- **R6-L1 / L2 / L3** 可作为下一批低危清理。

### 建议回贴给 AGENT 的返工要点

1. 统一 ONNX 计划口径：改 `PLAN.md:262,760` 与 `E10/P0/PLAN.md`，明确
   “ONNX 需要额外 onnx/onnxscript；默认不装，实际兜底为 .pt/.npz”。
2. `E0/code/run_all.py` 与 `E0/code/check_env.py` 改用 `src.constants` 的 train/test 常量；
   并把硬编码扫描回归扩展到这两个文件。
3. `bootstrap_data.sh`：
   - 显式 manifest 不存在必须非零退出；
   - manifest 缺 sha 字段必须非零退出；
   - `--verify-only` 不创建目录；
   - 修正或移除 `--dest` 在 tarball 模式下的不一致。
4. `docs/platform_setup.md:67` 的 `/v4_data/v4_data.tar.gz` 改为 `/data/v4_data.tar.gz`。
5. 云端实跑后回贴：
   - `python -c "import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
   - `pip freeze`
