"""TensorBoard 日志（对接平台集成）。

平台约定（训练任务文档 §TensorBoard 日志）：
    将日志写入环境变量 `TENSORBOARD_LOGDIR` 指向的目录，
    平台任务详情页即可看到"迭代曲线"。

关键点
------
1. `tensorboard` **不装在镜像里**（30 GB 磁盘 + 我们只做 JSON/CSV 日志），
   因此本模块用 try/except 探测；平台镜像通常自带 `torch.utils.tensorboard`，
   若不可用则**静默降级**为 JSONL（永远可写）。
2. `TENSORBOARD_LOGDIR` 由 `run_train.sh` 导出到 `$V4_DATA_ROOT/v4/tb`（**持久**，
   任务结束仍保留，符合"日志写 /data"的要求）。
3. 纪律不变：TensorBoard 只用于**观测**，模型选择/早停仍以 `src/score.py` 的
   真实分数为准（`资料库/12` §2.3）。

用法
----
    from src.training.tb_logger import RunLogger
    log = RunLogger("E3_unet_fold0")
    log.scalar("score/oof_total", 80.5, step=epoch)
    log.scalars({"loss/align": 0.31, "loss/aux": 0.12}, step=epoch)
    log.close()          # 或 with RunLogger(...) as log:
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """同时写 TensorBoard（若可用）与 JSONL（永远可用）。"""

    def __init__(self, run_name: str, logdir: str | Path | None = None,
                 enable_tb: bool | None = None) -> None:
        self.run_name = run_name
        base = logdir or os.environ.get("TENSORBOARD_LOGDIR") or "tb"
        self.dir = Path(base) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "scalars.jsonl"
        self._fh = self.jsonl.open("a", encoding="utf-8")
        self._writer = None
        if enable_tb is None:
            enable_tb = True
        if enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
                self._writer = SummaryWriter(log_dir=str(self.dir))
            except Exception:
                self._writer = None
        self.tb_available = self._writer is not None

    def scalar(self, tag: str, value: float, step: int | None = None) -> None:
        rec: dict[str, Any] = {"t": time.time(), "step": step, "tag": tag,
                               "value": float(value)}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), step)

    def scalars(self, values: dict[str, float], step: int | None = None) -> None:
        for k, v in values.items():
            self.scalar(k, v, step)

    def text(self, tag: str, body: str, step: int | None = None) -> None:
        if self._writer is not None:
            self._writer.add_text(tag, body, step)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def resolve_logdir() -> str:
    """返回本次任务的 TensorBoard 目录（优先平台变量，其次 /data，最后本地）。"""
    d = os.environ.get("TENSORBOARD_LOGDIR")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        return d
    data_root = os.environ.get("V4_DATA_ROOT", "/data")
    cand = Path(data_root) / "v4" / "tb"
    try:
        cand.mkdir(parents=True, exist_ok=True)
        return str(cand)
    except OSError:
        Path("tb").mkdir(exist_ok=True)
        return "tb"
