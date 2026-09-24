# 可复用训练成果持久化

> **文档导航**：[v4 文档中心](README.md) · [v4 README](../README.md) · [总计划](../PLAN.md) · [代码审查状态](CODE_REVIEW_STATUS.md)
> **文档类型**：跨任务复用与恢复手册。描述 `/data` 上到底保存什么、新机器如何恢复、以及如何避免重复训练。

## 0. 一句话

v4 采用**两层持久化**：

1. **小状态层** `tools/sync_state.py`：checkpoint、OOF、report、scaler → `/data/v4/mirror`
2. **大成果层** `tools/artifact_store.py`：特征缓存、折级 `.pkl`、提交包、候选/注册表 → `/data/v4/artifacts`

两者合起来，新开发机/新训练任务可以复用所有**有用的训练成果**，只重新生成真正缺失的本地数据与 cache。

## 1. 为什么不把所有东西都丢到 `/data`

`/data` 是 30 GB 网络盘，且平台训练任务本地盘会随任务结束丢失。如果每次都把整个 runtime 全量复制：

- 大 cache/checkpoint 会反复写，浪费时间；
- 会很快占满 30 GB 配额；
- 日志/TensorBoard/临时文件没有复用价值。

因此需要按“是否值得复用”分层。

## 2. 持久化清单

| 位置 | 持久化内容 | 工具 |
|---|---|---|
| `/data/v4/mirror/run_root` | `*.pt`、`*.pth`、`*.ckpt`、`*.npz`、`*.json`、`*.md`、`*.csv` | `sync_state.py` |
| `/data/v4/mirror/scalers` | scaler JSON | `sync_state.py` |
| `/data/v4/mirror/reports` | Gate / metrics / 报告 JSON | `sync_state.py` |
| `/data/v4/artifacts/cache/raw` | 原始分片 cache | `artifact_store.py` |
| `/data/v4/artifacts/cache/feat` | F2/win/well/phys 特征缓存 | `artifact_store.py` |
| `/data/v4/artifacts/run_root` | `runs/**/*.pkl`、`*.onnx`、`*.zip` | `artifact_store.py` |
| `/data/v4/artifacts/state` | `candidates.json`、`registry.json` 等 | `artifact_store.py` |
| `/data/v4/state` | 流水线进度、candidates、registry 快照 | `run_train.sh` |

明确不持久化：

- `data/*.txt` 原始井数据：从 `/data/v4_data.tar.gz` 重新部署即可；
- `logs/`、`tb/`、TensorBoard 事件：无复用价值；
- `*.tmp`、`__pycache__`、`.v4cache` 等临时内容。

## 3. 自动行为

`run_train.sh --mode all` 现在会自动：

1. 启动时：
   - 从 `/data/v4/state` 恢复进度；
   - 调用 `restore_state_from_network` 恢复小状态；
   - 调用 `restore_artifacts_from_network` 恢复大成果 cache / 折结果 / 提交包 / state JSON。
2. 每个 task 完成后：
   - 写入 `/data/v4/state` 进度；
   - 调用 `publish_artifacts_to_network` 增量发布大成果。
3. 任务/进程退出时：
   - `cleanup_state_mirror` 再同步一次小状态；
   - 再发布一次大成果，保证异常退出前已产出的内容也保存。

新机器上仍会重跑“本地缺失且无法复用”的步骤，例如：

- 原始数据部署（本地解压）；
- E0 raw cache 重建；
- 如果 `cache/feat` 已从 `/data/v4/artifacts` 恢复，则 E2 可跳过特征构建/消融；
- 如果 `runs/E3/.../fold_results/*.pkl` 已恢复，则 E3 可 fold 级续跑。

## 4. 手动命令

```bash
# 发布大成果到 /data/v4/artifacts
python3 tools/artifact_store.py publish \
  --local-root "$V4_DATA_ROOT" \
  --remote-root "$V4_NETWORK_ROOT"

# 从 /data 恢复大成果
python3 tools/artifact_store.py restore \
  --local-root "$V4_DATA_ROOT" \
  --remote-root "$V4_NETWORK_ROOT"

# 查看云端当前持有什么
python3 tools/artifact_store.py status \
  --local-root "$V4_DATA_ROOT" \
  --remote-root "$V4_NETWORK_ROOT"
```

`run_train.sh` 也提供手动模式：

```bash
# 手动发布小状态 + 大成果
bash run_train.sh --mode publish

# 手动恢复小状态 + 大成果
bash run_train.sh --mode restore
```

## 5. 预算与安全

- `artifact_store.py` 默认总大小上限 `24 GiB`；可用 `--max-gb 0` 关闭。
- 超过上限时程序会列出最大的文件并拒绝发布，不会静默写满 `/data`。
- 复制使用“同大小且 mtime 不旧则跳过”的增量策略。
- 所有写入先落临时文件再原子替换，避免复制中断留下半包。
- 需要暂时关闭自动大成果同步时：

```bash
export V4_DISABLE_ARTIFACT_STORE=1
export V4_ARTIFACT_MAX_GB=24
```

## 6. 新开发机恢复判断

一条命令检查 E0/E1/E2 是否已经缓存到 `/data`：

```bash
python3 tools/check_reuse_cache.py --remote-root /data
```

它会检查：

- E0 raw cache：`/data/v4/artifacts/cache/raw/{train,test}`（80/10 口）
- E1 OOF/checkpoint：`/data/v4/mirror/run_root/E1/`
- E2 feature cache：`/data/v4/artifacts/cache/feat/`
- E2 报告：`E2_ablation.json`、`E2_gate.json`、`E2_best_spec.json`
- 进度：`/data/v4/state/all_pipeline_progress.json` 的 task 1–5

也可以手动看：

```bash
ls -la /data/v4/state/
python3 tools/artifact_store.py status --remote-root /data
ls -la /data/v4/mirror/run_root/E1/
du -sh /data/v4/artifacts/cache/*
```

若命令输出 `RESULT: OK`，则 E0/E1/E2 可复用；若 `INCOMPLETE`，缺什么会逐项列出。
