# GraphLens

Static TFLite / LiteRT-LM graph analyzer for identifying graph-level blockers to QNN delegate partitioning.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## Scope

**Goals:**
1. Extract embedded .tflite from .litertlm
2. Parse graph structure (subgraphs, ops, tensors, dtypes, constness)
3. Check QNN builder availability and dynamic shape / index inputs
4. Simulate partition boundaries under these checks
5. Inspect seam context around fallback boundaries

**Non-goals:**
1. No LiteRT/QNN execution
2. No apply_plugin_main invocation
3. No runtime backend placement prediction
4. No guaranteed NPU support

## Quick Start

```python
from graphlens import Inspector

inspector = Inspector.from_litertlm(
    "model.litertlm",
    op_support_path="opSupportMap.csv"
)

inspector.analyze()
inspector.report_dynamic_shapes()
inspector.report_partitions()
inspector.report_seams()
```

## Setup

```bash
uv sync
uv run python examples/main.py
```

## Extract

Extract embedded TFLite flatbuffers from a `.litertlm` container using the existing heuristic scanner.

```python
from graphlens import Inspector

inspector = Inspector.from_litertlm(
    "model.litertlm",
    op_support_path="opSupportMap.csv",
    dump_dir="./dump",
)
```

![extract screenshot](docs/extract.png)

**Limitations:**
1. Root-offset and boundary detection are heuristic.
2. No schema/checksum validation is performed on extracted blobs.

## Parse

Parse subgraphs, ops, tensors, dtypes, and input constness from the selected `.tflite`.

```python
inspector = Inspector.from_tflite(
    "model.tflite",
    op_support_path="opSupportMap.csv",
)
inspector.analyze()
```

![parse screenshot](docs/parse.png)

**Limitations:**
1. `const_values` are decoded for INT32 only.
2. Operator attributes are not decoded.

## Dynamic shape detection

Identify shape/index inputs that are runtime-provided or inferred (`-1` in RESHAPE), which can cause static partition blockers.

```python
inspector.report_dynamic_shapes()
```

![dynamic shape screenshot](docs/dynamic_shape.png)

**Limitations:**
1. Detection is limited to `_SHAPE_INDEX_SLOTS` coverage.
2. `inferred_dim` detection is implemented for RESHAPE.

## Partition simulation

Simulate static NPU/CPU runs using two checks only: QNN builder availability and dynamic shape/index blockers.

```python
inspector.report_partitions()
```

![partition screenshot](docs/partition.png)

**Limitations:**
1. Does not model dtype-specific constraints, op attributes, or backend validation.
2. Results are static diagnostics, not runtime guarantees.

## Seam dump

Print context around partition boundaries to localize where fallback starts and ends.

```python
inspector.report_seams(context=2, kind="CPU")
```

![seams screenshot](docs/seams.png)

**Limitations:**
1. Context is based on flatbuffer op order.
2. Long partitions may be elided in the middle.

## Limitations

**Not modeled:**
1. Dtype-specific legalization constraints
2. Per-op attribute validation
3. Backend SDK validation/runtime heuristics
4. Hardware backend selection
