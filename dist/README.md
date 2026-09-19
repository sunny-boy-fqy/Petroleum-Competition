# dist/ 说明（数据分发包目录）

本目录**不进 git**（`*.tar.gz` 被 `.gitignore` 忽略），只作为本机/云盘之间的数据交换件。

## 生成

```bash
python3 v4/tools/pack_dataset.py
```

产物：

| 文件 | 大小 | 内容 |
|---|---:|---|
| `v4_data.tar.gz` | ≈30 MB | 90 口井原始 txt，内部路径 `v4/data/{train,test}/*.txt` |
| `v4_data_manifest.json` | ≈23 KB | 逐文件 sha256 + 行数 + 状态计数 + 非规范 schema 井清单 + 打包件 sha256 |

## 部署到平台云盘 `/data`

> **R5-B1 必读**：本目录不进 git（`dist/*.tar.gz`、`dist/*.json` 全被忽略），
> 所以云端从 Git 克隆出的 `/code/workspace/v4/dist/` 里**没有** tarball。
> 必须把 tarball 放到云盘，脚本会按以下顺序搜索：
> `--tarball` / `$V4_DATA_TARBALL` → `$V4/dist/v4_data.tar.gz` →
> **`$DATA_ROOT/v4_data.tar.gz`（云端推荐）** → `$DATA_ROOT/dist/v4_data.tar.gz`。

1. 把 `v4_data.tar.gz`（建议连带 `v4_data_manifest.json`）上传到云盘 **`/data/` 根目录**
   （或开发机的 `/data`）；
2. 在训练任务里执行：

```bash
bash /code/workspace/v4/run_train.sh --mode data
# 等价于（上传到 /data 根时无需任何参数）：
V4_DATA_ROOT=/data bash /code/workspace/v4/tools/bootstrap_data.sh
# 上传到别处时显式指定：
bash /code/workspace/v4/run_train.sh --mode data --tarball /data/uploads/v4_data.tar.gz
```

找不到 tarball 时会打印**搜索过的全部位置**并以 `exit 4` 退出，而不是静默失败。

部署后校验（应打印 `RESULT: OK`；80/10 井与 730,268/95,948 行是**无条件**硬校验，
manifest 存在时再加 tarball sha256）：

```
train wells=80 rows=730268  (expect 80 / 730268)
test  wells=10 rows=95948  (expect 10 / 95948)
  odd well 42f2870b: OK
  odd well b7eb1274: OK
  odd well c7611b01: OK
RESULT: OK
```

## 为什么不用平台「数据集挂载」

平台训练任务支持挂载数据集，但本赛题的原始井数据不在平台数据集广场里，
且挂载路径不如云盘 `/data` 可控。用「云盘 + 一次性解压」的方式：
数据只传一次、永久保留、路径固定为 `/data/v4/data`，与 `V4_DATA_ROOT` 契约一致。

若你确实把数据做成了平台数据集并挂载成功，可改用：

```bash
bash /code/workspace/v4/tools/bootstrap_data.sh --from-dir <挂载目录>
```
