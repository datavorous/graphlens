# tinygraphparser

Parse TFLite / LiteRT-LM graphs and detect statically visible blockers to QNN delegate partitioning (missing builders, non-constant shape/index inputs).

## Scope and non-goals

- This tool performs static analysis of TFLite flatbuffers only.
- It does not execute the model, invoke `apply_plugin_main`, or query the QNN SDK at any point.
- It cannot determine whether a specific op will be claimed by the delegate at runtime; it can only flag statically visible blockers.
- Real partitioning depends on factors outside this tool's scope: dtype-specific builder constraints, per-op attribute values, SDK version, and `backendValidateOpConfig` rejections.

## Setup

```
uv sync
uv run python examples/main.py
```

## Metrics produced

Every numeric or categorical output the tool currently emits, with definition and interpretation guidance.

| Metric | Definition | Interpretation guidance |
|--------|-----------|------------------------|
| **Op count per subgraph** | `len(sg["ops"])` for each subgraph in the parsed graph dict | Baseline denominator for all other per-subgraph metrics. A subgraph with very few ops is unlikely to contribute meaningful fragmentation signal. |
| **Op-type histogram (count, %, dtype split)** | Per-opname count, percentage of total ops, and per-output-dtype breakdown via `collections.Counter` | Surfaces the hot path and where mixed-precision occurs. Dominated by the wrong op type (e.g. many INT8 ops when expecting FLOAT32) is a signal worth investigating before partition analysis. |
| **Dynamic-shape candidate count** | Number of ops in `_SHAPE_INDEX_SLOTS` whose designated input slot is either non-constant or contains `-1` (RESHAPE only) | A high count suggests many ops may be ineligible for static NPU allocation. Does not confirm runtime behavior; only flags a statically visible condition. |
| **Partition count (NPU, CPU)** | Number of maximal contiguous runs classified as NPU-eligible or CPU-fallback by `simulate_partition` | Indicates the number of distinct static eligibility changes in the graph. A large number of alternating partitions signals frequent boundary transitions, which is a fragmentation concern under the modeled checks only. |
| **Largest / smallest / mean NPU partition size** | Op counts of NPU-classified `Partition` objects, computed in `report_partition` | Larger NPU runs suggest fewer delegate-boundary crossings under the modeled checks. Can mislead: a single large NPU partition can still be rejected at runtime by dtype or attribute constraints not modeled here. |
| **CPU fallback breakdown by reason** | Per-`reason` op count across all CPU partitions, grouped by `no_builder`, `dynamic_shape`, or `unsupported_composite` | Identifies which category of blocker drives fragmentation. Useful for prioritizing fixes: `no_builder` is a toolchain gap; `dynamic_shape` may be addressable by constant-folding. |
| **Agreement percentage** | `(total_ops - divergent - false_cpu) / total_ops` from `compare_to_actual` | Fraction of ops where static classification matches reported actual. Can be high while all ops of interest are misclassified, if the majority of ops are trivially eligible and agree trivially. |
| **Divergent op count** | Ops classified NPU by simulator but present in `actual["cpu_op_indices"]` | Lower-bounds the number of ops affected by factors outside this simulator's model. Does not identify which factor caused the discrepancy. |
| **False-CPU op count** | Ops classified CPU by simulator but absent from `actual["cpu_op_indices"]` | Ops the simulator flagged as ineligible that the runtime accepted anyway. Possible causes: the `opSupportMap.csv` is stale, or a dynamic shape input was resolved by the runtime in a way the static analysis couldn't see. |

---

## Extract

Scans the binary for `TFL3` magic bytes. For each hit, walks back up to 100 bytes to find a plausible flatbuffer root offset and slices out the blob. Writes each section as a separate `.tflite` file.

```python
from tinygraphparser import LiteRTLMExtractor

tflite_files = LiteRTLMExtractor.extract("model.litertlm", "./dump")
# ["./dump/Section1_TFLiteModel_heuristic.tflite", ...]
```

![extract screenshot](docs/extract.png)

### How it works

`LiteRTLMExtractor.extract` scans the raw binary for the 4-byte sequence `TFL3` (the TFLite file identifier). For each hit at position `pos`, it walks backward up to 100 bytes testing whether `content[test:test+4]` decoded as a little-endian uint32 falls in the range `(0, 1000)`, which is the heuristic for a plausible flatbuffer root offset. The slice boundary for the end of a blob is set to the next magic hit minus 100 bytes (or end-of-file). Blobs outside the size range `(1000, 10_000_000_000)` bytes are discarded.

### What it tells you

Which byte ranges in a `.litertlm` file are likely to be independent TFLite flatbuffers. This answers: "how many embedded models are in this container, and where do they start?" It does not identify the purpose or subgraph structure of each section; that requires parsing.

### Confidence

**SPECULATIVE.** The root offset heuristic (look-back up to 100 bytes, accept uint32 in `(0, 1000)`) is not derived from a documented `.litertlm` format specification. The end-of-blob boundary (`next_magic - 100`) may truncate or overlap blobs in files where adjacent sections are closer than 100 bytes.

### Known limitations

- The look-back window of 100 bytes and the root offset range `(0, 1000)` are arbitrary heuristics with no documented basis; they may fail on future format versions.
- End-of-blob trimming (`next_magic - 100`) can truncate trailing tensor data if two `TFL3` markers are within 100 bytes of each other.
- Corrupt or partial extractions produce flatbuffers with invalid tensor offsets near section boundaries; `TFLiteGraphParser` handles these with `_placeholder_tensor` fallbacks, but data in those slots is lost.
- No checksum or schema validation is performed on extracted blobs.

---

## Parse

Walks the flatbuffer: subgraphs → operators → tensors. For each input tensor, checks `model.Buffers(t.Buffer()).DataLength() > 0` to determine constness. INT32 constant buffers are decoded as little-endian int32 arrays and stored in `const_values`.

```python
from tinygraphparser import TFLiteGraphParser

graph = TFLiteGraphParser().parse(tflite_files[2])
# graph["subgraphs"][0]["ops"][0]["inputs"][1]["is_constant"]  -> True
# graph["subgraphs"][0]["ops"][0]["inputs"][1]["const_values"] -> [1, 128, 42]
```

Graph shape:
```python
{
  "path": str,
  "subgraphs": [{
    "name": str,
    "ops": [{
      "index":   int,
      "opname":  str,
      "inputs":  [{"name": str, "dtype": str, "shape": list,
                   "is_constant": bool, "const_values": list | None,
                   "tensor_index": int}],
      "outputs": [{"name": str, "dtype": str, "shape": list,
                   "tensor_index": int}],
    }]
  }]
}
```

![parse screenshot](docs/parse.png)

### How it works

`TFLiteGraphParser.parse` opens the file, calls `Model.GetRootAsModel` from the `tflite` flatbuffers package, then iterates `model.SubgraphsLength()` and `sg.OperatorsLength()`. For each operator, it looks up the opcode via `model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()` and maps it through a reverse enum dict built from `tflite.BuiltinOperator`. Input tensor constness is determined by `model.Buffers(t.Buffer()).DataLength() > 0`. INT32 buffers are decoded with `struct.unpack_from("<Ni", raw)`. Corrupt tensor offsets (common near heuristic extraction boundaries) are caught per-slot and replaced with `_placeholder_tensor(idx)`.

### What it tells you

The op sequence, tensor names, dtypes, shapes, and which input tensors have constant data embedded in the flatbuffer. This answers: "what is the computation graph structure, and which shape/index tensors are baked in versus supplied at runtime?" It does not decode non-INT32 constant buffers beyond flagging them as constant.

### Confidence

**ACCURATE** for the fields listed in the graph schema. The constness flag, dtype, shape, and opname are read directly from flatbuffer fields with no inference. `const_values` decoding is accurate for INT32 tensors; other dtypes return `const_values: None` even if the buffer is non-empty.

### Known limitations

- `const_values` is only decoded for `dtype == "INT32"`. FLOAT32 or INT64 constant buffers are flagged `is_constant: True` but `const_values: None`.
- Tensor shapes from heuristically extracted blobs may be zeroed or truncated at section boundaries; affected tensors are replaced by `_placeholder_tensor` with `shape: []` and `dtype: "UNKNOWN"`.
- Output tensors do not carry an `is_constant` field; they are assumed to be runtime-computed.
- Operator attributes (e.g. strides, padding mode, activation) are not decoded.

---

## Op histogram

Counts ops by opname using `collections.Counter`, then groups counts by `output[0].dtype` to surface where mixed-precision is happening. Results are sorted by total count descending.

```python
from tinygraphparser import report_op_histogram

report_op_histogram(graph)
report_op_histogram(graph, top=10)
```

![histogram screenshot](docs/histogram.png)

### How it works

`op_histogram` iterates every op across all subgraphs. It increments `totals[opname]` and `by_dtype[opname][dtype]` where `dtype` is `op["outputs"][0]["dtype"]` if outputs exist, otherwise `"UNKNOWN"`. The result list is produced by `totals.most_common()` with the per-dtype dict attached. `report_op_histogram` renders a bar scaled to the most frequent op, with a percentage and dtype split per row.

### What it tells you

Which op types dominate the graph, and at what output precision. This answers: "what is the computational character of this model, and where are dtype transitions happening?" For Gemma-family graphs, expect `MUL`, `ADD`, `FULLY_CONNECTED`, `RESHAPE`, and `SOFTMAX` near the top. Unexpected op types or dtype splits are worth investigating before partition analysis.

### Confidence

**ACCURATE.** Counts are exact; dtype is read from `output[0]["dtype"]`, which is a flatbuffer enum field. No inference is performed.

### Known limitations

- Dtype is taken from `output[0]` only. An op with heterogeneous output dtypes (e.g. a comparison op producing BOOL from FLOAT32 inputs) is counted under the dtype of its first output.
- Ops with no outputs are counted under `"UNKNOWN"` dtype.
- Histogram covers all subgraphs combined; per-subgraph breakdown requires calling `op_histogram` on a filtered graph dict.

---

## Dynamic shape detection

For each op in a fixed slot table (`_SHAPE_INDEX_SLOTS`), checks the input slot that carries shape or index data. Two failure modes: `runtime` (the tensor has no backing buffer so the shape is only known at runtime) and `inferred_dim` (the buffer exists but contains `-1`, RESHAPE only, meaning the shape is partially computed). Both prevent static memory layout resolution on the NPU.

```python
from tinygraphparser import report_dynamic_shape_ops

report_dynamic_shape_ops(graph)
```

Checked ops: `RESHAPE`, `PAD`, `PADV2`, `MIRROR_PAD`, `STRIDED_SLICE`, `SLICE`, `GATHER`, `GATHER_ND`, `SCATTER_ND`, `BROADCAST_TO`, `TILE`, `TRANSPOSE`, `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR`

![dynamic shape screenshot](docs/dynamic_shape.png)

### How it works

`find_dynamic_shape_ops` iterates ops whose opname is a key in `_SHAPE_INDEX_SLOTS`, a hard-coded dict mapping opname to a list of `(slot_index, label)` pairs. For each listed slot, it reads `op["inputs"][slot]["is_constant"]`. If `False`, it appends a `runtime` entry. If `True` and the opname is `RESHAPE` and `-1` appears in `const_values`, it appends an `inferred_dim` entry. All other constant slots pass silently. The function returns a list of dicts; `report_dynamic_shape_ops` formats and prints them with optional name clipping (`max_name`) and row cap (`max_hits`).

### What it tells you

Which ops have shape or index inputs that are not fully resolved at compile time, under the constraints this tool checks. This narrows the set of fragmentation candidates: an op appearing here is a statically visible reason it may not be placed on the NPU. It does not confirm the op will be rejected at runtime, nor does it cover all reasons an op might be rejected.

### Confidence

**PARTIAL.** The check is accurate for the listed slots of the listed ops. What is checked: whether `DataLength() > 0` for the designated input slot, and whether the INT32 values contain `-1` for RESHAPE. What is not checked: non-INT32 constant buffers for inferred dims; ops outside `_SHAPE_INDEX_SLOTS`; ops where the shape-carrying slot index differs from the table entry.

### Known limitations

- `_SHAPE_INDEX_SLOTS` is a manually maintained table. Ops not in the table are silently skipped even if they have shape-carrying inputs.
- `inferred_dim` detection is only implemented for `RESHAPE`. Other ops with constant shape buffers containing `-1` are not flagged.
- `EXPAND_DIMS` is in the table but is not listed in the README op list above; check the source if coverage matters.
- A slot flagged `runtime` by this tool may be constant-folded by a downstream compiler pass; this tool has no visibility into compiler transformations.
- `_placeholder_tensor` entries (from corrupt extraction boundaries) have `is_constant: False` and will be reported as `runtime`, which may be a false positive.

---

## Partition simulation

For each op, eligibility = `opname in opSupportMap` AND `op not flagged by find_dynamic_shape_ops`. Ops are walked linearly; a new partition is flushed whenever eligibility changes or the CPU fallback reason changes (so `no_builder` and `dynamic_shape` runs are never merged). This identifies statically visible fragmentation candidates. Real partitioning also depends on dtype, attribute, and SDK-level checks not modeled here.

```python
from tinygraphparser import load_op_support, simulate_partition, report_partitions

op_support = load_op_support("analysis/opSupportMap.csv")
partitions = simulate_partition(graph, op_support)
report_partitions(partitions)
```

CPU fallback reasons: `no_builder` · `dynamic_shape` · `unsupported_composite`

![partition screenshot](docs/partition.png)

### How it works

`load_op_support` reads the tab-separated `opSupportMap.csv`, strips the `kLiteRtOpCodeTfl` prefix from each op code, and converts the remainder from CamelCase to `UPPER_SNAKE_CASE` via `_camel_to_upper_snake` (with a manual override table `_NAME_OVERRIDES` for irregular names). The resulting set is stored in `OpSupport.tfl_supported`. `simulate_partition` calls `find_dynamic_shape_ops` to build a per-subgraph set of ineligible op indices, then walks ops linearly via `_partition_subgraph`. A new `Partition` is flushed whenever `eligible` changes or, for CPU runs, `reason` changes. `_classify_op` returns `(True, None)` for NPU-eligible ops, or `(False, reason)` for `no_builder`, `dynamic_shape`, or `unsupported_composite`.

### What it tells you

Which contiguous op runs are blocked by a missing builder or a non-constant shape/index input, under the two checks this tool performs. This answers: "given only builder availability and dynamic-shape inputs, how many eligibility boundaries exist in this graph?" It narrows the set of possible fragmentation sources; it does not confirm the actual partition structure the runtime will produce.

### Confidence

**PARTIAL.** What is checked: builder presence in `opSupportMap.csv`; dynamic-shape inputs as defined in `_SHAPE_INDEX_SLOTS`. What is not checked: dtype-specific builder constraints; per-op attribute values; `backendValidateOpConfig` rejections; composite op name matching beyond the presence of any `STABLEHLO_COMPOSITE`/`SHLO_COMPOSITE` entry.

### Known limitations

- `opSupportMap.csv` must be kept in sync with the SDK version under test; a stale CSV produces incorrect eligibility classifications with no warning.
- The CamelCase-to-UPPER_SNAKE_CASE converter may mis-map op names not in `_NAME_OVERRIDES`; a silently wrong mapping causes a false `no_builder` classification.
- Composite op eligibility is coarser than real dispatch: if any `SHLO_COMPOSITE` entry exists in the CSV, all `STABLEHLO_COMPOSITE` and `SHLO_COMPOSITE` ops are marked eligible, regardless of the composite's specific name.
- Linear walk assumes the op ordering in the flatbuffer reflects execution order; ops reordered by a compiler pass would not be reflected.

---

## Seam dump

Resolves each CPU partition's `op_indices` back to positions in the flat ops list, then prints `context` ops before and after the partition body. Use this to see exactly what op caused a split without re-running the full analysis.

```python
from tinygraphparser import report_seams

report_seams(graph, partitions, context=2, kind="CPU")
```

![seams screenshot](docs/seams.png)

### How it works

`report_seams` builds a position index `pos_by_idx = {op["index"]: list_position}` for each subgraph. For each partition of the requested `kind`, it resolves `p.op_indices[0]` and `p.op_indices[1]` through `pos_by_idx` to get list positions `start_pos` and `end_pos`. It then prints ops at positions `[start_pos - context, start_pos)` (labeled `prev`), the partition body (labeled `>>>`), and `(end_pos, end_pos + context]` (labeled `next`). Long partition bodies are elided to the first and last `context` ops.

### What it tells you

The exact ops that flank each CPU-fallback boundary, under the static classification. This answers: "what is the last NPU-eligible op before the fallback, and what is the first NPU-eligible op after it?" Useful for manually investigating why a specific op was classified as a blocker without re-running the full simulation.

### Confidence

**ACCURATE** relative to the partition simulation output. The seam dump faithfully renders what `simulate_partition` produced; it introduces no additional inference. Its accuracy as a description of real runtime boundaries inherits the PARTIAL confidence of `simulate_partition`.

### Known limitations

- Seam context is based on list position in the flatbuffer op array, not execution order; if ops were reordered by a compiler pass, the displayed neighbors may not be the true runtime neighbors.
- The elision (`... N more in this partition ...`) hides interior ops; a blocker embedded deep inside a long CPU partition will not be visible at the default `context=2`.
- Only one `kind` can be displayed per call; call twice (with `kind="NPU"` and `kind="CPU"`) to see both.

---

## Predicted vs actual

Expands each partition's `op_indices` range into a set of individual op indices, then intersects with `actual["cpu_op_indices"]`. `divergent_ops` = predicted NPU but actually CPU; these indicate factors not modeled by this simulator. `agreement` is the fraction of ops where both verdicts match.

```python
from tinygraphparser import compare_to_actual, report_comparison

actual = {
    "main": {
        "npu_partitions": 4,
        "cpu_partitions": 3,
        "cpu_op_indices": [488, 891, 1450],
    }
}
diffs = compare_to_actual(partitions, actual)
report_comparison(diffs)
```

![comparison screenshot](docs/comparison.png)

### How it works

`compare_to_actual` expands each `Partition.op_indices` range `(start, end)` into the integer set `{start, start+1, ..., end}`, accumulating into `predicted_npu` and `predicted_cpu` sets. It then reads `actual[subgraph]["cpu_op_indices"]` as `actual_cpu` and derives `actual_npu = all_ops - actual_cpu`. `divergent_ops = predicted_npu & actual_cpu` (simulator said NPU, runtime said CPU). `false_cpu_ops = predicted_cpu & actual_npu` (simulator said CPU, runtime said NPU). `agreement = (total - |divergent| - |false_cpu|) / total`.

### What it tells you

The size and location of the gap between static classification and reported runtime behavior. `divergent_ops` lower-bounds the number of ops affected by factors this simulator does not model; it does not identify which factor. `agreement` gives an overall calibration score for the simulator against a specific runtime result, but can be high while all interesting ops are misclassified if most ops are trivially eligible.

### Confidence

**PARTIAL.** The arithmetic is exact given the inputs. Confidence in the result depends entirely on the accuracy of `actual["cpu_op_indices"]`; if that data comes from an approximation or a different graph version, the comparison is meaningless. The op-index expansion assumes `op_indices` ranges are contiguous in the flatbuffer ordering, which matches the simulator's assumptions but may not match runtime execution groupings.

### Known limitations

- `actual["cpu_op_indices"]` must be sourced from the same flatbuffer version and the same SDK run that produced the partition under test; any mismatch silently produces wrong agreement numbers.
- Expansion of `(start, end)` to a contiguous range assumes no gaps in op indices within a partition; if the flatbuffer has non-contiguous op numbering, the expanded set will include indices that do not correspond to real ops.
- Partition counts (`npu_partitions`, `cpu_partitions`) in `actual` are stored but not used in agreement computation; they are printed for reference only.
- The comparison does not identify which specific unmodeled factor (dtype, attribute, SDK version) caused each divergent op.
