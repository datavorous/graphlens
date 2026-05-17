# LensRT

TFLite graph analyzer for QNN delegation diagnostics.

## Install

```bash
git clone git@github.com:datavorous/LensRT.git
cd LensRT && uv sync
```

Static analysis works after this. For runtime analysis see [Runtime setup](#runtime-setup) below.

## Usage

### Static

```python
from lensrt import Static

s = Static("models/qwen.tflite", "datasets/opSupportMap.csv")
s.report()
data = s.json()
```

For `.litertlm` files:

```python
s = Static.from_litertlm("model.litertlm", "datasets/opSupportMap.csv")
```

Expected output (first signature only):

```
Rank violations
No rank violations found in models/qwen.tflite

Dynamic shape / index inputs
4 fragmentation candidate(s) in models/qwen.tflite
  [  17] GATHER_ND  (subgraph: decode)
         slot 1 (indices): RUNTIME  ai_edge_torch.generative.utilities.co...  INT64  [1, 1]
  [  54] RESHAPE  (subgraph: decode)
         slot 1 (new_shape): RUNTIME  arith.constant204  INT32  [0]

Cross-signature divergence
1 cross-signature divergence candidate(s) in models/qwen.tflite
  [  17] GATHER_ND  appears in 2 subgraph(s)
         decode
         prefill_128

Partition simulation
Signature: decode (1326 ops)
  Partitions: 5 total, 3 delegated, 2 non_delegated
  Largest delegated partition: 1271 ops (ops 55-1325)
  Smallest delegated partition: 17 ops (ops 0-16)
  Mean delegated partition: 441.3 ops
  Non-delegated breakdown:
    dynamic_shape: 2 ops [GATHER_ND x1, RESHAPE x1]
```

### Runtime

```python
from lensrt import Runtime

r = Runtime(
    model="models/qwen.tflite",
    plugin="path/to/libLiteRtCompilerPlugin_Qualcomm.so",
    tool="path/to/apply_plugin_main",
    soc="SM8650",
    qnn_lib="path/to/qairt/lib/x86_64-linux-clang",
    out="./runtime_out",
)
r.report()
data = r.json()
```

Expected output:

```
Signature: decode
  Non-delegated original ops: 125
  Top non-delegated op types:
      121  FULLY_CONNECTED
        1  EMBEDDING_LOOKUP
        1  GREATER_EQUAL
        1  LESS_EQUAL
        1  GATHER_ND

Signature: prefill_128
  Non-delegated original ops: 168
  Top non-delegated op types:
      116  FULLY_CONNECTED
       48  DYNAMIC_UPDATE_SLICE
        1  EMBEDDING_LOOKUP
        1  GREATER_EQUAL
        1  LESS_EQUAL

Global (log-derived)  -- not attributed to any subgraph
  ValidateOp rejection codes:
    3110  dtype_mismatch        586
  Top rejected op types:
      474  FullyConnected
       96  ElementWiseSelect
        8  ElementWiseBinary
        4  Gather
        4  GatherNd
```

Outputs written to `out=`:
- `rewritten.tflite` — flatbuffer with DISPATCH_OP nodes
- `run.log` — apply_plugin_main log

## What it checks

### Static (no SDK required)
- Missing QNN builder
- Input rank exceeds QNN cap
- Dynamic shape or index input
- Inferred -1 dim (RESHAPE, PAD, BROADCAST_TO, TILE)
- Cross-signature divergence
- Dtype risk (FLOAT32 on FC/Gather — delegated only if quantized)

### Runtime (requires LiteRT build + QNN SDK)
- Per-subgraph delegated/non_delegated op counts from rewritten flatbuffer
- Global ValidateOp rejection codes from log (not per-subgraph)

## Runtime setup

Runtime needs LiteRT built from source plus the QNN SDK. Skip this if static is enough.

### 1. LiteRT + bazel

```bash
git clone https://github.com/google-ai-edge/LiteRT.git
# bazelisk auto-fetches bazel 7.7.0 from .bazelversion
pip install bazelisk
```

### 2. QNN SDK

Download Qualcomm AI Runtime Community v2.46.0.260424 from
softwarecenter.qualcomm.com (Qualcomm account required). Unzip so the
layout is `qairt/2.46.0.260424/{bin,include,lib,...}`.

### 3. Make QAIRT a bazel workspace

```bash
cp LiteRT/third_party/qairt/qairt.BUILD qairt/2.46.0.260424/BUILD
echo 'workspace(name = "qairt")' > qairt/2.46.0.260424/WORKSPACE
```

### 4. Build the binaries

```bash
cd LiteRT
QAIRT=/absolute/path/to/qairt/2.46.0.260424

bazel build //litert/tools:apply_plugin_main \
  --override_repository=qairt=$QAIRT

bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin \
  --override_repository=qairt=$QAIRT
```

First build is 15-30 min. Outputs:

- `LiteRT/bazel-bin/litert/tools/apply_plugin_main`
- `LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so`

### 5. System libs

Arch:

```bash
sudo pacman -Sy libc++
# only if libunwind.so.1 is missing
ln -sf /usr/lib/libunwind.so.8 /usr/lib/libunwind.so.1
```

Ubuntu / Debian:

```bash
sudo apt install libc++-dev libc++abi-dev libunwind-dev
```

Verify (should print nothing):

```bash
ldd LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so | grep "not found"
```

### 6. Plug paths into Runtime()

```python
Runtime(
    model="path/to/model.tflite",
    tool="LiteRT/bazel-bin/litert/tools/apply_plugin_main",
    plugin="LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so",
    qnn_lib="qairt/2.46.0.260424/lib/x86_64-linux-clang",
    soc="SM8650",
    out="./runtime_out",
)
```

## Common SoC strings

| Device | `soc=` |
|---|---|
| Snapdragon 8 Gen 3 | SM8650 |
| Snapdragon 8 Gen 2 | SM8550 |
| Snapdragon 888 | SM8350 |

## Limitations
- Static checks are necessary but not sufficient for delegation
- Runtime error codes are log-global, not per-subgraph
- Delegated = DISPATCH_OP present; actual backend (HTP/HVX/GPU) determined by QNN runtime, not this tool
- opSupportMap.csv pinned to LiteRT commit b2df679f
