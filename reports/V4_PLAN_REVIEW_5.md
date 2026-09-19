# V4 五审报告

> 审查对象：`014f616`（四审返工）与 `1dce27e`（环境事实 CUDA 12.8 / torch 2.7.1 + 依赖清单，R5）
> 审查方式：直接读代码 / 独立重跑 / 构造反例 / 干净 `git archive` 复跑
> 结论：**五审不通过**。四审的 5 组回归确实闭环，R5 的 `check_env` 旧硬断言也已正确拆除；但云端数据部署路径、空数据泄漏检查、依赖文档三处仍会误导或阻塞实际流程。

---

## 0. 已独立复核通过的部分

| 检查 | 结果 |
|---|---|
| `tests/run_all.py`（torch 2.4.0+cpu 本机校验 venv） | 195 项，0 fail，0 skip |
| `tests/run_all.py`（无 torch） | 195 项，0 fail，37 skip |
| 四审点名 5 组回归 | 36 项全过：E0 prereg 复算、`build_model(init_stats=...)`、SW 反例、真实 forward→decode、`disk_guard` 七路径 |
| `tools/check_data_leak.py`（有数据时） | 90 井，违规 0，RESULT: OK |
| `tools/check_consistency.py` | 候选 1，错误 0，RESULT: OK |
| `tools/check_status.py` | 阶段 12 / P 33，RESULT: OK |
| `tools/plan_stats.py --check` | 46 份 / 6,843 行，Gate 13/13，RESULT: OK |
| `tools/gen_data_card_md.py --check` | RESULT: OK |
| `tools/sync_prereg_templates.py --check` | 33 份，RESULT: OK |
| `predict.py --use-version CONST` | 10 井 / 95,948 行，contract_ok |
| 干净 `git archive` 解包 | 190 项，0 fail，9 skip；文档检查表面 OK（但见 R5-H1） |

**四审核心阻断项的复核结论：**

- `check_env` 已从“硬断言具体 runtime 12.4/12.6”改为“hard 只判 major==12，声明值与驱动能力分别 warn/advisory”，四审 R4-B1 的 `exit 11` 根因已移除。
- `RowMLP.forward` 已显式分键：`q_atom/q_joint` 是概率，`q_atom_logit/q_joint_logit` 是 logits；`total_loss` 只接受 logits，`decode_predictions` 对只给 logits 的 dict 做一次 sigmoid；真实链路测试已覆盖。
- SW 契约已从“全体 median”升级为“低值计数 + 低于有效最小占比 + 非原子行 p05 + median”四重守卫，7/10 原子行 + 3 行 0.8 的反例会被拒绝。
- E0 prereg/report 已统一为 13/13，`aggregate_gate` 复算通过，`reports/E0_gate_result.json` 落盘。
- `E0_plan_stats.json` 已由 `sync_plan_stats.py` 同源重生成，并被 `plan_stats --check` / `check_status.py` 逐字段校验。
- `disk_guard` 的 `cleanup→save_and_exit` 已统一走 `_escalate`，hook 会被调用；`abort` 不再被 `allow_soft` 软接受。

---

## 1. 高危 / 会阻塞推荐流程的问题

### R5-B1. 云端 `run_train.sh --mode data` 找不到用户上传到云盘的 tarball

**位置**

- `run_train.sh:131-135`：
  ```bash
  run_data() {
    log "--- [data] 部署数据集到 $DATA_ROOT/v4/data"
    bash "$HERE/tools/bootstrap_data.sh" 2>&1 | tee -a "$LOG"
  ```
  `run_data` **没有传任何 tarball 路径**，也没有消费 `EXTRA_ARGS`。
- `tools/bootstrap_data.sh:23-24`：
  ```bash
  TARBALL="$V4/dist/v4_data.tar.gz"
  MANIFEST="$V4/dist/v4_data_manifest.json"
  ```
  脚本只会在 `/code/workspace/v4/dist/` 下找文件。
- `docs/platform_setup.md:94-95`、`docs/training_tasks.md:35-41`、`dist/README.md:20-24` 都告诉用户：
  > 把 `v4_data.tar.gz` 上传到**云盘 `/data`**，然后执行
  > `bash /code/workspace/v4/run_train.sh --mode data`
- 但 `.gitignore` 明确忽略 `dist/*.tar.gz` / `dist/*.json`，云端从 Git 仓库克隆出来的 `/code/workspace/v4/dist/` **不可能**有这个 tarball。

**独立复现（干净 git archive 解包，无 dist tarball）：**

```text
$ V4_DATA_ROOT=/tmp/r5data2 bash tools/bootstrap_data.sh
== v4 bootstrap_data ==
repo       : /tmp/v4_archive_r5
data root  : /tmp/r5data2
dest       : /tmp/r5data2/v4/data
!! 既没有 --from-dir 也没有 /tmp/v4_archive_r5/dist/v4_data.tar.gz
   请先在本机运行: python3 v4/tools/pack_dataset.py
$ echo $?
4
```

因此按文档执行 `env → data → e0` 时，**第二步 data 会直接退出 4**；云端 E0 无法完成，第五步之后也无从谈起。这不是“验证没做”，而是推荐路径本身不可执行。

**修复方向**

- `bootstrap_data.sh` 增加 `--tarball <path>` 和 `$V4_DATA_TARBALL` 支持；
- 默认搜索路径至少包括：`$V4/dist/v4_data.tar.gz`、`$V4_DATA_ROOT/v4_data.tar.gz`、`$V4_DATA_ROOT/dist/v4_data.tar.gz`；
- `run_data` 把 `EXTRA_ARGS` 透传给 `bootstrap_data.sh`，或显式传 `--tarball`；
- manifest 也一并搜索并校验；docs 的“上传到云盘”示例和实际路径保持一致；
- 增加 shell 级回归：无 tarball -> 非零退出；`--tarball` 指向云盘 -> 解压并校验通过。

---

### R5-H1. `tools/check_data_leak.py` 在零输入井时输出 `RESULT: OK`，回归检查是空转

**位置**：`tools/check_data_leak.py`

```python
checked = 0
for split, with_targets in (("train", True), ("test", False)):
    for f in sorted((root / split).glob("*.txt")):
        ...
        checked += 1
...
print("RESULT:", "OK" if not violations else "FAIL")
return 0 if not violations else 1
```

`checked` 从未拿 90（或至少 >0）做判据。如果数据路径错误、目录为空、或干净 `git archive` 解包后没有 `../data`，脚本会打印：

```text
检查井数: 0（train+test），输入列数要求: 13
违规: 0
RESULT: OK
returncode=0
```

这是一条**假绿**：一个用于防止“80 口井全部标签泄漏”的回归，在没有任何井时反而宣告通过。用户报告中的“干净 `git archive` 解包复跑同样全绿”之所以包含这条，也是因为它当时空转。

**修复方向**

- 显式要求 `checked == 80 + 10`（或至少 train=80 / test=10 分别断言）；
- `--data` 不存在 / 为空时返回非零；
- 增加单测：临时空目录必须 FAIL，90 井目录必须 PASS。

---

## 2. 中优先级问题

### R5-H2. 云端 `bootstrap_data.sh` 往往拿不到 manifest，于是训练行数不被校验

**位置**：`tools/bootstrap_data.sh:80-90`

```python
rows_tr = sum(...)
rows_te = sum(...)
ok = (n_tr == 80 and n_te == 10 and rows_te == 95948)
if man_path.is_file():
    ...
    ok = ok and (rows_tr == c["n_train_rows"])
```

`dist/v4_data_manifest.json` 和 tarball 一起被 `.gitignore` 忽略。docs 只要求上传 `v4_data.tar.gz`，因此云端很可能没有 manifest：

- tarball sha256 不校验；
- `rows_tr == 730268` **不校验**；
- 仅测试集 95,948 行仍会被检查。

一个截断/损坏但恰好保有 80/10 个文件和 95,948 条测试行的 tarball 仍可通过。建议：

- `ok = (n_tr == 80 and n_te == 10 and rows_tr == 730268 and rows_te == 95948)` 无条件成立；
- manifest 存在时再额外校验 sha + 精确计数；
- 把 `v4_data_manifest.json`（~23 KB）纳入 git，或要求与 tarball 一起上传并在 `--tarball` 旁自动搜索。

### R5-M1. `PLAN.md` 仍保留旧的 pip/onnx/pyarrow/tensorboard 口径

R5 已把 required 依赖改成 `numpy/pandas/scipy/scikit-learn/einops`，并明确 pyarrow/onnx/onnxruntime 不需要、tensorboard 可选推荐。但 `PLAN.md`（项目自称唯一权威）仍有冲突段落：

- `PLAN.md:208-209`：
  ```bash
  python -m pip install --no-cache-dir \
      "pandas>=2.0,<3" "pyarrow>=15" "scipy>=1.11" "scikit-learn>=1.4" \
      "einops>=0.7" "onnx>=1.16" "onnxruntime>=1.17"
  ```
- `PLAN.md:279`：预算表仍写“额外 pip 包（pandas/pyarrow/scipy/sklearn/einops/onnx/onnxruntime...）”，控制手段写“**不装 tensorboard**”。
- `PLAN.md:321`：明确写“不安装：tensorboard/...”（与 `PLAN.md:219` 的“推荐 tensorboard”矛盾）。
- `docs/image_requirements.md:88-99` 的 Dockerfile 追加层仍同时出现：
  ```dockerfile
  RUN python -m pip install --no-cache-dir \
        "numpy" "pandas" "scipy" "scikit-learn" "einops" \
        "scipy==1.13.1" "scikit-learn==1.5.2" "einops==0.8.0" \
        "onnx==1.16.2" "onnxruntime==1.18.1" \
   && python -m pip cache purge
  ```
  既有重复钉死版本，又把 R5 明确“不需要”的 onnx/onnxruntime 装进镜像；文本又推荐 tensorboard，而 Dockerfile 没装。

这些文档会直接和 `requirements.txt` / `setup_deps.sh` / `docs/dependencies.md` 打架。建议按 R5 的唯一清单重写 PLAN §3.3.1、§3.4.1 和 image_requirements Dockerfile 方案 B。

### R5-M2. `setup_deps.sh` 的 lock 路径会跳过 numpy，和 required 清单不一致

`check_env.REQUIRED_PY_DEPS`、`requirements.txt`、用户 pip 清单都包含 `numpy`，但 `setup_deps.sh`：

```bash
grep -vE '^\s*#|^\s*$|^torch|^numpy' "$LOCK" | ...
```

从 lock 中显式排除了 `numpy`。如果平台镜像确实有 numpy，这不是问题；但代码/脚本的契约与“required”声明不一致：一旦镜像升级去掉 numpy，`--mode env` 的自动安装不会补它，随后 full 校验会 hard fail。建议二选一：

- 在 `setup_deps.sh` 正常路径也安装 numpy（版本不钉死）；或
- 在 `check_env` 中把 numpy 标为“镜像保证，不通过 pip 补充”，并用一个更明确的 `IMAGE_PROVIDED_DEPS` 集合表达，而不是混在 REQUIRED 里。

---

## 3. 低优先级问题

### R5-L1. `docs/image_requirements.md` 的期望输出把 warn-ok 行写成 `[WARN]`

`check_env.py` 的打印规则是：

```python
mark = "OK  " if c["ok"] else ("FAIL" if c["level"] == "hard" else "WARN")
```

因此 `cuda_runtime_declared`（level=warn、ok=true）和 `cuda_driver_version`（level=warn、ok=true）实际会打印 `[OK  ]`，不是 `[WARN]`。但 `docs/image_requirements.md:123-125` 的“期望输出”写作：

```text
[WARN] cuda_runtime_declared   ...
[WARN] cuda_driver_version     ...
```

这会让用户误以为 `--mode env` 不正常。应改为 `[OK]`，或把打印规则改成按 level 显示但文档明确说明“level=warn 即使通过也显示 WARN”。

### R5-L2. `constants.py` 的列注释仍写“14 输入曲线”

`src/constants.py:13` 的注释：

```python
"DEVI", "AZIM", "BIT", "CASE",  # 14 输入曲线（DEPTH 为深度基准）
```

`COLUMNS` 是 17 列规范布局，`INPUT_COLUMNS` 是 13 条曲线；DEPTH 不是曲线。R4-M5 修了 `parse.py` / `features/basic.py`，这条注释仍容易误导 E2/E3 工程师。建议改成“14 个输入字段 = DEPTH + 13 条曲线”。

### R5-L3. `bootstrap_data.sh` 的 tar 路径注释与实际不一致

`tools/bootstrap_data.sh:67` 注释：

```text
# 解压到 DATA_ROOT，tar 内路径为 data/train/... 与 data/test/...
```

实际 `dist/v4_data.tar.gz` 内路径是 `v4/data/train/...`。当前 `tar -C "$DATA_ROOT"` 恰好正确，但如果后人按注释手工调整，会解压到错误位置。改成 `v4/data/...` 即可。

---

## 4. 建议的返工顺序

1. **修 R5-B1**：先让 `env → data → e0` 在云端真正可执行，否则后面所有 Gate 都是空谈。
2. **修 R5-H1**：让 `check_data_leak.py` 在 0 井时 FAIL，并加入空数据单测。
3. **修 R5-H2**：无条件校验 train=730268 / test=95948，manifest 存在时再 sha 校验。
4. **修 R5-M1/M2**：统一 PLAN、image_requirements、requirements.txt、setup_deps.sh 的依赖事实。
5. 修 R5-L1/L2/L3 文档小项。
6. 重跑：
   ```bash
   .venv-torch/bin/python tests/run_all.py
   ../v2/.venv/bin/python tests/run_all.py
   .venv-torch/bin/python tools/check_data_leak.py
   .venv-torch/bin/python tools/check_consistency.py
   .venv-torch/bin/python tools/check_status.py
   .venv-torch/bin/python tools/plan_stats.py --check
   .venv-torch/bin/python tools/gen_data_card_md.py --check
   .venv-torch/bin/python tools/sync_prereg_templates.py --check
   .venv-torch/bin/python predict.py --use-version CONST --data_dir ../data/test --output /tmp/r5_const.json
   ```
   并新增：
   - 干净 archive 中 `tools/check_data_leak.py --data <空目录>` 必须非零；
   - `bash tools/bootstrap_data.sh --tarball <云盘路径>` 必须解压并校验成功；
   - `run_train.sh --mode data` 在“tarball 只在云盘”的布局下必须成功；
   - docs / PLAN 依赖事实一致性检查。

---

## 5. 五审结论

四审提出的 3 个高危阻断项（CUDA runtime 语义、logit/probability 错位、SW median 守卫可绕过）**确认已闭环**；
R5 对环境事实的修正方向也正确。

但本轮新增/遗留了会直接卡住云端流程的一处硬问题：**`run_train.sh --mode data` 依靠 repo 内的 `dist/v4_data.tar.gz`，而文档和云盘持久化策略都默认 tarball 在 `/data`**。同时 `check_data_leak.py` 的 0 井假绿和 PLAN/docs 依赖口径冲突需要收口。

**五审结论：不通过；修完 R5-B1 / R5-H1 / R5-H2 / R5-M1 后可进入六审或直接作为云端实跑前置。**
