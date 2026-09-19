# V4 四审报告

> 审查对象：`bf2ed47..HEAD`（`1125569` + `26c0411`）
> 审查方式：不采信返工报告结论，直接读代码、重跑验收命令、构造反例
> 结论：**四审不通过**。工程收口明显进步，但存在 3 个会直接破坏云端 Gate / 推理门控 / SW 契约的高危问题，外加若干计划-代码漂移与证据过期。

---

## 0. 已独立复核通过的部分

| 项 | 结果 |
|---|---|
| `tests/run_all.py`（torch 2.4.0+cpu） | 132 tests，0 fail，0 skip |
| `tests/run_all.py`（无 torch） | 132 tests，0 fail，29 skip |
| `tools/check_data_leak.py` | 90 井，违规 0，RESULT: OK |
| `tools/check_consistency.py` | 候选 1，错误 0，RESULT: OK |
| `tools/check_status.py` | 阶段 12 / P 33，RESULT: OK |
| `tools/plan_stats.py --check PLAN.md` | 46 份 / 6,781 行，RESULT: OK |
| `tools/gen_data_card_md.py --check` | RESULT: OK |
| `tools/sync_prereg_templates.py --check` | 33 份，RESULT: OK |
| `predict.py --use-version CONST --data_dir ../data/test` | 10 井 / 95,948 行，contract_ok=true |
| `git archive HEAD` 干净解包复跑 | 127 tests，0 fail，9 skip |
| Gate 方向修复（E9/P2、E10/P0/P1） | 抽查通过；`max_degradation=0.01 → PASS`，`0.35 → FAIL`，缺指标 `→ FAIL` |
| `.gitignore` 根锚定修复 | `src/models/` 已入 git；根级 `models/`、`cache/` 等仍忽略 |
| 数据/缓存主证据 | 80/730,268、10/95,948、32.39 MB、drop=70.490735、mask=69.843218 均可复算 |

因此，问题不是“没做事”，而是**仍有少量阻断性硬点没有闭环**。

---

## 1. 高危阻断项

### R4-B1. `check_env.py` 用 `torch.version.cuda`（12.4）硬比 CUDA 12.6，云端 Gate 必挂

**位置**

- `E0/code/check_env.py:39`：
  ```python
  EXPECTED_CUDA_MAJOR_MINOR = (12, 6)
  ```
- `E0/code/check_env.py:137-142`：
  ```python
  cuda_ver = getattr(torch.version, "cuda", None)
  want_cuda = "12.6"
  if cuda_ver:
      got_mm = ".".join(str(cuda_ver).split(".")[:2])
      rep.add("cuda_version", got_mm == want_cuda, level,
              f"torch.version.cuda={cuda_ver} (expected {want_cuda} driver)")
  ```

**问题**

平台镜像锁死为 `torch==2.4.0+cu124`（`versions/locks/cloud.txt`、`docs/dependencies.md`、`PLAN.md §3.3.1`）。
`+cu124` wheel 的 `torch.version.cuda` 是 **12.4**，不是 12.6；CUDA 12.6 是**驱动能力**，不是 PyTorch 运行时版本。

因此 `run_train.sh --mode env` 中 `check_env_profile base` 会得到：

```
[FAIL] cuda_version  torch.version.cuda=12.4 (expected 12.6 driver)
hard failures: 3
```

`run_env` 随即 `exit 11`，不会继续执行 disk guard，也就永远无法产出通过云端 Gate 所需的 `E0_env.json` / `E0_disk_budget.json`。

这与 `E0/P0/PLAN.md:38`“逐项核对 torch/cuda/gpu/bf16/disk 五行”直接冲突。

**修复方向**

- 要么把 hard 断言改为 `torch.version.cuda == 12.4`，文档注明“12.4 runtime + 12.6 driver”；
- 要么单独用 `nvidia-smi` 查询 driver 版本（若平台能提供稳定的 driver 查询），不要拿 `torch.version.cuda` 冒充 CUDA 12.6；
- 同步更新 `docs/image_requirements.md` 的期望输出和 JSON 的 `expected.cuda`。

---

### R4-B2. 模型输出 `q_atom`/`q_joint` 是 logits，但原子门与解码接口按“概率”使用

**位置**

- `src/models/row_mlp.py:102-115`：
  ```python
  q_joint = self.q_joint(h).squeeze(-1)
  q_atom = torch.stack([...], dim=1)
  return {
      "por": por,
      "perm_z": perm_z,
      "sw": sw,
      "q_atom": q_atom,       # 未过 sigmoid，是 logit
      "q_joint": q_joint,     # 未过 sigmoid，是 logit
      "ph_logit": q_joint,
  }
  ```
- `src/inference/atomic_gate.py:83-84`：
  ```python
  q_atom: (N,3) 原子概率
  命中条件为严格大于 q_atom[:,t] > tau[t]
  ```
- `src/features/basic.py:272-283`：
  ```python
  q_atom = out.get("q_atom", q_atom)
  ...
  out = per_target_hard_switch(out, np.asarray(q_atom, dtype="float64"), tau_atom, atom_values)
  ```
- `src/features/basic.py:289-290`：
  ```python
  out = joint_guard(out, np.asarray(q_joint, dtype="float64"), tau_high, atom_values)
  ```

**问题**

`RowMLP` 返回的是 **logits**（`tests/test_heads.py:83-84` 明确用 `torch.sigmoid(out["q_joint"])` / `torch.sigmoid(out["q_atom"])` 验证先验）。
但 `per_target_hard_switch`、`joint_guard`、`select_tau_per_target`、`misclassification_cost_report` 全部按 **概率** 解释输入，`E6/P1` 又明确在 `[0.05, 0.95]` 上搜概率阈值。

这会导致：

- `q_atom` 的 `0.05 ~ 0.95` 阈值搜索实际只覆盖 `sigmoid([0.05,0.95]) ≈ [0.512,0.721]` 的概率区间；
- 真正最优概率阈值若在 0.1 或 0.9，当前搜索根本到不了；
- 更严重的是，调用方如果直接使用 `row_mlp` 输出，会把 logits 与概率阈值混用，产生系统性误判。

**修复方向**

二选一，并加集成测试：

1. **推荐**：`forward` 同时返回
   - `q_atom_logit`/`q_joint_logit` 供 BCE 损失；
   - `q_atom = torch.sigmoid(q_atom_logit)`、`q_joint = torch.sigmoid(q_joint_logit)` 供门控；
   - `ph_logit` 保留为 `q_joint_logit` 别名。
   然后 `total_loss` 改用 `*_logit`。
2. 或者保持输出 logits，但在 `decode_predictions`、`select_tau_per_target`、`joint_guard` 入口强制 `sigmoid`，并把所有文档/Tests 改成“输入 logit”。

无论哪种，必须新增“真实 `RowMLP.forward` 输出 → `decode_predictions` → 硬切换”的链路测试；当前测试全部用手工构造的概率矩阵，所以完全没暴露这个接口错位。

---

### R4-B3. SW 尺度契约的 median 守卫可被 66.7% 原子行绕过

**位置**

- `src/inference/contract.py:204-220`：
  ```python
  scales = row_scales or {"SW": (C.SW_MIN_OBSERVED, 100.0), ...}
  ...
  med = [float(p.get("SW", 0.0)) for ...]
  m = _st.median(med)
  ...
  if m < lo:
      res.ok = False
      res.errors.append("SW median ...")
  ```

**反例（已实测）**

构造 10 行预测：7 行 SW=99.9（联合占位），3 行 SW=0.8（被错误归一化到 [0,1] 的连续分支）。
整体中位数 = 99.9 > 8.305，契约 **passed**：

```
ok True
errors []
sw median 99.9
```

这正是 E6/PD1 架构最可能出现的情况：**66.7% 原子行把 median 拉到 99.9，掩盖剩余连续分支的 [0,1] 归一化错误**。
而且 `E0/code/run_all.py:54-58` 的 `good_minimal` 契约正样例里也用了 `SW=0.4`——按“单一百分数、有效最小 8.305”的事实，它本身就不应是正样例。

**修复方向**

- 不再使用全体 median 作为唯一 SW 尺度守卫；
- 增加低值计数或低分位守卫，例如：
  - `n(SW < 1) / n_obs > 0.001` → FAIL；
  - 或 `p05/p25 < SW_VALID_MIN` → FAIL（但要注意 atom 占比过高时 p25 仍可能偏高，低值计数更直接）；
- 低值阈值建议用实测事实 `SW<1=0` 和 `SW_VALID_MIN=8.305` 双重判据；
- 修改 `run_all.py` 的 `good_minimal` 正样例，把 `SW=0.4` 改成真实尺度值（如 80.0）；
- 增加“原子行占多数 + 连续行被归一化”的回归测试。

---

## 2. 高优先级一致性问题

### R4-H1. `E0_gate_prereg.json` 与 `E0_local_contract_gate.json` 不能通过同一个 Gate 校验器复算

**复现实测**

```python
pr = json.load(open("reports/E0_gate_prereg.json"))
rep = json.load(open("reports/E0_local_contract_gate.json"))
aggregate_gate(pr, {"checks": rep["mandatory_checks"], "abs_diff": 0.0})
```

结果：

```
passed=False
mandatory_failures=['contract_ok']
absolute_checks={'abs_tolerance': ... 'ok': True ...}
```

原因：

- `reports/E0_gate_prereg.json` 的 `mandatory_checks` 有 **13 项**，包含 `contract_ok`；
- `reports/E0_local_contract_gate.json` 的实际 checks 只有 **12 项**，包含 `contract_selftest`，**没有 `contract_ok`**；
- prereg 的 `primary_threshold_key=abs_tolerance`，但 report 里没有 `abs_diff/score_diff` 字段，实际用 `aggregate_gate` 复算时该项默认失败（缺指标即失败）。
- `tools/check_status.py` 只比较 status 与 report 的 12 项，不校验 prereg 与 report 的可复算一致性。

**修复方向**

- 把 `contract_ok` 作为 `contract_selftest` 的显式别名写入 local gate report（13/13），或删除 prereg 中多余项并在 `mandatory_exempt` 说明；
- 给 report 增加 `abs_diff`（E0 可复算为 0.0），或移除 `abs_tolerance` 的绝对门槛；
- 增加回归测试：`aggregate_gate(E0_gate_prereg, E0_local_contract_gate_result)["passed"] is True`。

---

### R4-H2. `reports/E0_plan_stats.json` 严重过期

**实测**

```
reports/E0_plan_stats.json:
  main=730, stage=708, p_level=4988, total=6426

tools/plan_stats.py measure():
  main=808, stage=754, p_level=5219, total=6781
```

`PLAN.md`、`README.md`、`docs/PROJECT_FILES.md`、`versions/status.json` 都已同步到 6,781，但 `reports/E0_plan_stats.json` 仍是旧值。
原因是最终改动后没有重新运行 `tools/plan_stats.py --json reports/E0_plan_stats.json`，而 `tools/check_status.py` 和 `tests/test_plan_stats.py` 都不检查该 JSON 与 `measure()` 的一致性。

**修复方向**

- 重生成 `reports/E0_plan_stats.json`；
- 在 `tools/check_status.py` 或 `tools/plan_stats.py --check` 中增加“JSON 证据与 measure() 完全相等”的断言；
- 把这个 JSON 纳入 E0 回拷/证据清单，避免再次成为“已提交但过期”的手写副本。

---

### R4-H3. `disk_guard` 在 cleanup 后进入 save_and_exit 时不调用 `capacity_hook`，且 `allow_soft` 可覆盖 abort

**位置**：`src/data/disk_guard.py:224-242`

**问题**

当前 cleanup 分支：

```python
if st.level == "cleanup":
    st = cleanup()
    st = disk_state(...)
    if st.level == "ok": return st
    if allow_soft: return st          # ← 明确写着“仍低于 min_gb”，但未排除 abort
    raise DiskBudgetError(...)        # ← 即使已经是 save_and_exit，也只抛错，不调用 capacity_hook
```

已用 monkeypatch 复现：初始 `cleanup`、cleanup 后 `save_and_exit`，`capacity_hook` 调用次数为 **0**。
这违反模块 docstring 的“free < 5 GB → 先回调 capacity_hook 保存 last.pt，再抛错”。同时 `allow_soft` 的当前位置可能把 `abort` 也软接受，违反“free < 3 GB 绝不继续写盘”。

**修复方向**

- cleanup 后重新测量，并对结果统一走同一套 `save_and_exit` / `abort` 分支：
  - `save_and_exit` → 调 hook，再抛错；
  - `abort` → 不调 hook，立即抛错，且不受 `allow_soft` 影响；
- 增加 `test_disk_guard.py`，用 monkeypatch 序列覆盖 `ok / cleanup→ok / cleanup→save_and_exit / cleanup→abort / allow_soft` 五种路径。

---

## 3. 中优先级问题

### R4-M1. `fit_target_scalers` 的返回字典不能直接喂给 `build_model(init_stats=...)`

**实测**

```python
sc = fit_target_scalers(...)
build_model(32, init_stats=sc)
# TypeError: RowMLP.init_from_stats() got an unexpected keyword argument 's_por'
```

`fit_target_scalers` 返回 `{por_median, por_max, sw_mu, sw_sigma, perm_z_median, s_por, s_sw}`，
但 `build_model` 会把整个 dict 转发给 `RowMLP.init_from_stats`，而后者不接受 `s_por/s_sw`。
E1/P1 训练脚本几乎必然会遇到这个接口错位。

**修复方向**：`build_model` 只转发 `RowMLP.init_from_stats` 签名内的键；或给 `init_from_stats` 增加 `s_por=None, s_sw=None` 并忽略/存入 buffer；并加集成测试。

### R4-M2. E6 计划与 `src/inference/atomic_gate.py` 漂移

| 计划写法 | 代码/实现 |
|---|---|
| `E6/P0/PLAN.md:32,90`、`docs/gen_p_details.py:1029,1068` 写 `src/models/state_head.py` | 实际五个头在 `src/models/row_mlp.py`，不存在 `state_head.py` |
| `E6/P1/PLAN.md:52` 和生成器写 `q ≥ τ` | `atomic_gate.py:84` 与 `PLAN.md:540` 都是严格 `q > τ` |
| `E6/P1/PLAN.md:38,49` 写网格 `[0.05,0.95]` 步长 `0.01`（91 点） | `select_tau_per_target` 默认 `np.linspace(0.05,0.95,19)`（19 点，步长 0.05） |
| `E6/P1/PLAN.md:51` 写平台容差 `ε=1e-3` | `select_tau_per_target` 默认 `tol=1e-4` |

其中网格步长最重要：默认调用会得到 0.05 分辨率，而不是计划声明的 0.01，会直接影响 `τ_t` 选择；`tests/test_atomic_gate.py:151` 还把“19 点”锁进了测试，导致计划与测试也互相矛盾。

**修复方向**：确定唯一口径（建议 code 默认 0.01 / 91 点、tol=1e-3；计划硬切换改为严格 `>`；E6 代码归属改为 `row_mlp.py` 或新建 `state_head.py` 并让 generator 同源）。

### R4-M3. `aggregate_gate` 的 `absolute` 分支没有使用指标字段映射

**位置**：`src/validation/gates.py` `aggregate_gate` 的 `gtype == "absolute"` 分支：

```python
key = prereg["primary_threshold_key"]
if key not in th or score is None:
    metric_pass = False
...
metric_pass = (score >= want) if direction == "min" else (score <= want)
```

它永远读 `result["score"]`，即使 `primary_threshold_key` 是 `min_auc`、`max_minutes` 等应映射到 `auc`/`minutes` 的键。
当前 33 份模板没有 `gate_type=absolute`，所以未暴露；但 Gate 类型已声明支持 `absolute`，这是一个确定的 latent bug。

**修复方向**：absolute 分支也用 `resolve_metric_value(key, result)`；补 `gate_type=absolute` + `min_auc`/`max_minutes` 的测试。

### R4-M4. `τ_t` 单测用的是 0/1 容差准确率，不是官方 soft score

`tests/test_atomic_gate.py` 的 `_official_acc_por/perm/sw` 返回：
```python
mean(err < 1.0)
```
而 `src/score.py` 的官方 Acc 是：
```python
mean(clip(1 - err, 0, 1))
```
两者对 `τ` 排序不一定一致。`select_tau_per_target` 本身是 generic 的，但单测以“官方目标函数”命名，实际只测了二值准确率；生产 E6 若照抄测试 helper，会优化错目标。

**修复方向**：单测直接调用 `src/score.py` 的逐目标分数包装；至少加一个“同一组数据下 soft-score 与 0/1 准确率选出不同 τ”的测试，确保生产路径使用的是 `score.py`。

### R4-M5. 特征/解析 docstring 仍写 14 列输入

- `src/data/parse.py:51`：`inputs: Any  # (n, 14) 规范顺序`
- `src/features/basic.py:51-52`：`inputs : (n, 14) / missing: (n, 14)`

但 E0-R2 后实际是 `inputs (n,13)` / `missing (n,13)`，`depth` 单独一列；`build_row_features` 也硬断言 13。
这类 stale docstring 会误导 E2/E3 的工程师把 14 当输入通道。

### R4-M6. E0 缓存输入列校验只检查 train shard

`E0/code/run_all.py --with-cache` 的校验循环：

```python
for w in DS.iter_wells(cache_root, "train"):
    ...
```

`shard_cache_input_cols_ok` 只覆盖 80 个训练 shard，不含 10 个 test shard。
虽然 `full_90_wells` 泄漏回归会检查 `n_inputs=13`，但“缓存本身”的输入列完整性声明并不等价，建议两个 split 都查。

---

## 4. 低优先级文档/证据问题

- `README.md:145` 标题写“E0 已经查出的**两个**硬事实”，下面实际列了 6 条。
- `E0/PLAN.md:90` 写“所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`”，但 E0 实际 report 没有任何一个，靠 `mandatory_exempt` 豁免；这句话需要限定为“所有模型 Gate”。
- `E0_plan_stats.json` 已列在 R4-H2，不重复。
- `docs/image_requirements.md:107-114` 的“期望输出”没有列出 `cuda_version`，无法帮助发现 R4-B1。

---

## 5. 四审建议的返工顺序

1. **先修 R4-B1**：否则云端 `--mode env` 永远无法点灯，E0 云端 Gate 无法闭环。
2. **修 R4-B2**：明确 logit/probability 契约，改 forward + loss + inference + tests。
3. **修 R4-B3**：把 SW 契约从“中位数守卫”升级为“低值计数/低分位 + 正例尺度修正”。
4. **修 R4-H1/H2/H3**：Gate prereg 复算、plan_stats 证据同步、disk_guard 安全路径。
5. **修 R4-M1..M6**：接口/计划/测试漂移。
6. 最后重跑：
   ```bash
   .venv-torch/bin/python tests/run_all.py
   ../v2/.venv/bin/python tests/run_all.py
   .venv-torch/bin/python tools/check_data_leak.py
   .venv-torch/bin/python tools/check_consistency.py
   .venv-torch/bin/python tools/check_status.py
   .venv-torch/bin/python tools/plan_stats.py --check PLAN.md
   .venv-torch/bin/python tools/gen_data_card_md.py --check
   .venv-torch/bin/python tools/sync_prereg_templates.py --check
   .venv-torch/bin/python predict.py --use-version CONST --data_dir ../data/test --output /tmp/r4_const.json
   ```
   并新增/通过：
   - `aggregate_gate(E0_gate_prereg, E0_local_contract_gate_result) == passed`
   - `build_model(init_stats=fit_target_scalers(...))`
   - “原子占多数 + 连续 SW 归一化”反例被契约拒绝
   - “真实 forward + decode_predictions/ha”集成测试
   - disk_guard 五路径单测

---

## 6. 结论

本轮不是“没修好”，而是**工程修复没有覆盖到真实云端镜像语义、真实 forward 输出语义、以及占位主导分布下的契约边界**。

- 四审结论：**不通过**；
- 最优先阻断：`check_env` CUDA runtime 误判、`q_atom/q_joint` logits/probability 错位、SW median 守卫可绕过；
- 其余为证据/计划/接口一致性问题，修完即可进入五审。
