# E4 Patch Transformer 与多尺度融合

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E4/src/、v4/E4/code/`　|　产物 `v4/experiments/E4/、v4/models/E4/`

> 阶段性质：**第二主干与融合阶段。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

实现 PatchTST 式通道独立 Patch Transformer，并与 CNN 主干做多尺度融合（门控或 concat），评估是否超过单一 CNN 主干。

## 2. 为什么这么设计

1. `资料库/08` §0.3 第 3 层与 §7.5 指出 Patch 化 + 通道独立是长序列建模的低成本高效方案。
2. CNN 主干擅长局部形态，Transformer 擅长长程依赖；两者并联是常见且有效的互补结构。
3. 必须先有 E3 的 CNN 结果做对照，否则无法判断 Transformer 是否值得其算力成本。

## 3. 输入

- E3 主干与 OOF
- `资料库/08` §7（注意力细节）

## 4. 产物

- `src/models/patchtf.py`、`src/models/multiscale.py`
- `models/E4/**`、`experiments/E4/**/oof.npz`
- `reports/E4_gate.json`、`reports/E4_param_budget.json`

## 5. 代码归属

- `E4/code/train_patchtf.py`、`E4/code/fuse_multiscale.py`

## 6. P 级子计划

- [P0 Patch Transformer 主干](P0/PLAN.md)
- [P1 多尺度融合](P1/PLAN.md)

## 7. 完成判据（Gate）

- 多尺度或 PatchTF 至少一个配置 ≥ E3 最佳（CI 下界 > 0），否则明确 NO-GO 并保留 E3 结构。
- 参数量/显存/单折耗时报告齐备。

## 8. 禁止事项

- 用 2.5+ 的注意力 API
- 引入需要编译的注意力扩展（flash-attn/xformers）

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
