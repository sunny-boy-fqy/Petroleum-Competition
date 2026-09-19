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

1. 把 `v4_data.tar.gz` 上传到云盘（或开发机的 `/data`）；
2. 在训练任务里执行：

```bash
bash /code/workspace/v4/run_train.sh --mode data
# 等价于：
V4_DATA_ROOT=/data bash /code/workspace/v4/tools/bootstrap_data.sh
```

部署后校验（应打印 `RESULT: OK`）：

```
train wells=80 rows=730268
test  wells=10 rows=95948
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
