"""可选依赖探测与降级层。

即便 pip 可用，也必须保留本层：本机（开发机）可能没有 torch / pandas / pyarrow，
评测机与本机也可能不同。所有可选库统一从这里获取，缺失即走兜底路径。

用法
----
    from portability import HAS_TORCH, HAS_PANDAS, HAVE, save_columns, load_columns
    if HAS_PANDAS: ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------- 探测
def _probe(name: str) -> tuple[bool, str | None]:
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", "unknown")
    except Exception:  # ImportError 及其它（如版本不兼容的 RuntimeError）
        return False, None


HAS_NUMPY, NUMPY_VERSION = _probe("numpy")
HAS_PANDAS, PANDAS_VERSION = _probe("pandas")
HAS_PYARROW, PYARROW_VERSION = _probe("pyarrow")
HAS_SCIPY, SCIPY_VERSION = _probe("scipy")
HAS_SKLEARN, SKLEARN_VERSION = _probe("sklearn")
HAS_TORCH, TORCH_VERSION = _probe("torch")
HAS_ONNX, ONNX_VERSION = _probe("onnx")
HAS_ONNXRUNTIME, ONNXRUNTIME_VERSION = _probe("onnxruntime")
HAS_EINOPS, EINOPS_VERSION = _probe("einops")

HAVE: dict[str, bool] = {
    "numpy": HAS_NUMPY,
    "pandas": HAS_PANDAS,
    "pyarrow": HAS_PYARROW,
    "scipy": HAS_SCIPY,
    "sklearn": HAS_SKLEARN,
    "torch": HAS_TORCH,
    "onnx": HAS_ONNX,
    "onnxruntime": HAS_ONNXRUNTIME,
    "einops": HAS_EINOPS,
}

VERSIONS: dict[str, str | None] = {
    "numpy": NUMPY_VERSION,
    "pandas": PANDAS_VERSION,
    "pyarrow": PYARROW_VERSION,
    "scipy": SCIPY_VERSION,
    "sklearn": SKLEARN_VERSION,
    "torch": TORCH_VERSION,
    "onnx": ONNX_VERSION,
    "onnxruntime": ONNXRUNTIME_VERSION,
    "einops": EINOPS_VERSION,
}


def describe() -> dict[str, Any]:
    return {"available": HAVE, "versions": VERSIONS}


def require(*names: str) -> None:
    """硬依赖断言：缺失时抛出带修复建议的错误（不允许静默）。"""
    missing = [n for n in names if not HAVE.get(n, False)]
    if missing:
        raise RuntimeError(
            f"v4 requires {missing} but they are not importable. "
            "基础栈（torch/numpy）应由镜像预装，不得 pip 替换；"
            "额外轻量包请运行 v4/E0/code/setup_deps.sh（--no-cache-dir）。"
        )


# ---------------------------------------------------------------- 列式存储降级
def save_columns(path: str | Path, columns: dict[str, Sequence[Any]]) -> str:
    """保存一维/二维列数据。

    首选 parquet（若 pyarrow 可用），否则回退 np.savez_compressed，
    并把列名顺序写入同名 `.json` 以便无损还原。

    返回实际使用的格式名："parquet" | "npz" | "json"。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: list(v) for k, v in columns.items()}

    if HAS_PANDAS and HAS_PYARROW:
        import pandas as pd  # noqa: PLC0415

        out = p.with_suffix(".parquet")
        pd.DataFrame(cols).to_parquet(out, index=False)
        return "parquet"

    if HAS_NUMPY:
        import numpy as np  # noqa: PLC0415

        out = p.with_suffix(".npz")
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        out.with_suffix(".npz.json").write_text(
            json.dumps({"columns": list(cols.keys()), "format": "npz"},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return "npz"

    out = p.with_suffix(".json")
    out.write_text(json.dumps(cols, ensure_ascii=False), encoding="utf-8")
    return "json"


def load_columns(path: str | Path) -> dict[str, list[Any]]:
    """与 save_columns 对应的读取（自动识别 parquet / npz / json）。"""
    p = Path(path)
    for cand, kind in ((p.with_suffix(".parquet"), "parquet"),
                       (p.with_suffix(".npz"), "npz"),
                       (p.with_suffix(".json"), "json")):
        if not cand.is_file():
            continue
        if kind == "parquet":
            import pandas as pd  # noqa: PLC0415

            df = pd.read_parquet(cand)
            return {c: df[c].tolist() for c in df.columns}
        if kind == "npz":
            import numpy as np  # noqa: PLC0415

            meta = cand.with_suffix(".npz.json")
            names = (
                json.loads(meta.read_text(encoding="utf-8"))["columns"]
                if meta.is_file()
                else None
            )
            with np.load(cand, allow_pickle=False) as z:
                keys = names or list(z.files)
                return {k: z[k].tolist() for k in keys}
        return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no saved columns found near {p}")
