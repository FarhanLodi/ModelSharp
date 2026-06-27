# B5 — GPU decoder kernels, on-device KV-cache & LLM-on-GPU validation (current state)

## Summary of what changed in this pass

The GPU engine (`IlgpuEngine`) now dispatches the full transformer **compute** path on CUDA, plus a
persistent **on-device KV-cache** with a stateful decode seam. Validated on an RTX 4090 against
`ManagedCpuEngine` to ~1e-3 (`GpuLlmTests.cs`, `GpuCudaParityTests.cs`).

## Data-movement behaviour (unchanged from the prior pass, still true)

`IlgpuEngine.Run` keeps **all intermediate float tensors resident on the device** across the op chain:
inputs/initializers upload once, every op writes a fresh device output buffer, and there is a single
`_accelerator.Synchronize()` immediately before the graph outputs are read back. One device→host copy
per graph output, not per op. Integer/bool tensors (token ids, masks, shape/index vectors) flow
host-side as dtype-carrying `Tensor` values, which is correct and pragmatic for the index math they feed.

## New GPU ops added this pass (each with a hardware-CUDA parity test)

Audited against the real `distilgpt2.onnx` (1569 nodes, 30 distinct ops). The decoder ops that were
missing and are now implemented on the GPU engine:

| Op            | Device path                                                                              |
|---------------|-------------------------------------------------------------------------------------------|
| `Reshape`     | shape-only; float buffer copied device→device, int stays host                            |
| `Unsqueeze`   | shape-only (same as Reshape)                                                              |
| `Squeeze`     | shape-only (same as Reshape)                                                              |
| `Shape`       | emits dims as a host int64 tensor (pure metadata)                                         |
| `Constant`    | float value → device, int value → host                                                    |
| `Expand`      | float broadcast via the device Gather kernel (host-precomputed offsets); int host-side    |
| `Split`       | float chunks sliced on-device via the Gather kernel; int host-side                        |
| `Pow`         | new `PowK` broadcasting kernel (device)                                                   |
| `Where`       | new `WhereK` broadcasting kernel (device, condition uploaded as 0/1 floats); int host     |
| `Erf`         | new `ErfK` device kernel (A&S 7.1.26, matching the CPU `MathHelpers.Erf`)                 |
| `Gemm`        | device: optional A/B transpose (Transpose kernel) → MatMul kernel → α/β scale + C add     |

All of these are exercised by `GpuLlmTests.cs` (`Cuda_Pow_*`, `Cuda_Erf`, `Cuda_Where_*`,
`Cuda_Reshape_*`, `Cuda_Unsqueeze_Squeeze_*`, `Cuda_Expand_*`, `Cuda_Split_*`, `Cuda_Gemm_*`), each
asserting CUDA-vs-CPU parity to 1e-3 and confirming `IsHardwareGpu==true`, skipping cleanly when no CUDA.

`Run`'s `finally` now dedupes buffers by reference before disposing, so unwinding mid-graph (e.g. on an
unsupported op) can't double-free a device buffer.

## distilgpt2 GPU-op audit (see `GpuDistilGpt2AuditTests` and `GpuLlmTests.DistilGpt2_Gpu_Coverage_Is_Complete`)

- **1569 / 1569 nodes (100%) are GPU-dispatchable** — the engine's `Run` switch now covers every distinct
  op type in distilgpt2. No CPU fallback remains.
- The mask / position-id prologue ops that previously fell back — `Range`, `ConstantOfShape`, `Equal`,
  `Greater`, `Trilu`, `ScatterND` — are now dispatched **host-side** (see "Integer/mask prologue ops"
  below). They are integer/boolean control-flow ops that build the causal mask and position ids and are
  never on the float compute path, so computing them on the host-resident int/bool tensors (and uploading
  any rare float result via `Load`) keeps GPU-vs-CPU parity exact while keeping the design's host/device
  split. Each is covered by a hardware-CUDA parity test in `GpuPrologueOpsTests.cs`.
- The transformer **compute** path itself (Gemm/MatMul/Softmax/LayerNorm/attention, the Pow+Tanh GELU, and
  all the Reshape/Transpose/Split/Concat/Gather/Slice plumbing) is GPU-complete and proven by the
  attention-block + multi-step KV-cache tests.

## Integer/mask prologue ops added this pass (each with a hardware-CUDA parity test)

| Op                | Host-side computation (output placement)                                              |
|-------------------|---------------------------------------------------------------------------------------|
| `Range`           | `start..limit` step `delta`, dtype follows inputs (int → host, float → device)        |
| `ConstantOfShape` | shape from 1-D int input, fill from `value` attr; float fill → device, int/bool → host |
| `Equal`           | same-dtype broadcast `==`, Boolean output (host)                                       |
| `Greater`         | same-dtype broadcast `>` (IEEE NaN semantics on float), Boolean output (host)          |
| `Trilu`           | upper/lower triangle with diagonal `k`, batched, dtype-preserving (float → device)     |
| `ScatterND`       | copy-then-write index tuples (`batch_dims=0`, `reduction=none`), dtype-preserving      |

All six mirror the ONNX semantics of their CPU kernels exactly, so a whole-graph distilgpt2 GPU run no
longer stops at `/transformer/Range`.

## What now runs ENTIRELY on GPU end-to-end

- A full **scaled-dot-product self-attention block** (`Transpose → MatMul → Mul(scale) → Softmax →
  MatMul`) runs entirely on the GPU engine and matches the CPU engine to 1e-3
  (`GpuLlmTests.Cuda_Attention_Block_Matches_Cpu`).
- A **multi-step autoregressive decode** over the on-device KV-cache (5 steps) runs entirely on GPU and
  matches a CPU full-attention reference at every step to 1e-3
  (`GpuLlmTests.Cuda_OnDevice_KvCache_MultiStep_Matches_Cpu`).
- Not the full distilgpt2 graph — see the fallback list above.

## On-device KV-cache (new)

`GpuKvCache` (in `GpuKvCache.cs`) owns two device buffers laid out `[numHeads, maxSeq, headDim]` for K
and V that **persist across decode steps**. It is created via `IlgpuEngine.CreateKvCache(numHeads,
maxSeq, headDim)` (borrows the engine's accelerator) and tracks the current `SeqLen`; `Reset()` reuses
it for a new sequence; `Dispose()` frees the buffers.

The stateful decode seam is `IlgpuEngine.DecodeStepAttention(cache, stepQ, stepK, stepV, scale?)`:
1. **Append** the step's per-head K/V into the persistent cache at the current `SeqLen` offset via a
   device→device `SubView.CopyTo` — **no realloc, no host round-trip**; `SeqLen` advances by `stepLen`.
2. Compute attention of the step's query against the **entire cached prefix** on-device: per head it
   builds Kᵀ (Transpose kernel), `scores = Q·Kᵀ` (MatMul kernel), scales (broadcast Mul), `softmax`
   (Softmax kernel), then `ctx = attn·V` (MatMul kernel). Only the final context is downloaded.

This does **not** disturb the existing stateless `IExecutionEngine.Run` contract — it is an additional
public surface on the engine.

## B5 completion pass — full decoder layer through the cache seam + seq-0 fix + whole-graph e2e

This pass closes the two gaps the previous "honest list" called out.

### 1. seq-0 / zero-length device buffers (whole-graph blocker) — FIXED

Driving the WHOLE distilgpt2 graph through `Run` failed at prefill because the empty `past_key_values.*`
float inputs are shape `[1, heads, 0, head_dim]` → **0 elements**, and `ILGPU.Allocate1D(emptyArray)`
yields a 0-length buffer whose `View`/`SubView`/`CopyTo` fault on later use (Concat with the new K, then
MatMul). The fix:

- A single allocation chokepoint, `AllocFloat(long len)`, allocates `max(len,1)` so a length-0 float tensor
  is backed by a **sentinel 1-element buffer** while the carried `TensorShape` still records length 0. This
  extends the over-allocate/trim convention `DeviceValue.ToHost` already documented to *every* float output.
- `Load` represents an empty float input via that sentinel (no `Allocate1D(emptyArray)`).
- Every device kernel launch is **guarded on the real length** (`if (n > 0) …`) — added to the binary
  fast/broadcast paths, unary, LeakyRelu, Transpose/Transpose2D, Softmax, Reduce (incl. identity copy),
  LayerNorm, MatMul, Gemm (all three internal passes) — so a 0-element op writes nothing and the empty past
  contributes nothing to Concat/Gather/MatMul (a contraction over an empty past sums to 0, the ONNX-correct
  result). The Gather/Slice/Expand/Split/Pow/Where paths already used the `n==0?1:n` + `if (n>0)` idiom.
- `RunConcat`'s float re-upload now routes through `Load` (so an empty concat result is sentinel-backed too).

Proven by `GpuWholeGraphTests.Cuda_EmptyPast_Concat_Then_MatMul_Matches_Cpu`: an empty `past` `[H,0,D]` is
loaded, Concat'd with this step's K along the sequence axis, transposed and MatMul'd into attention scores —
no crash — and matches the CPU engine; the same graph then runs with a non-empty past to confirm the decode
(append) path keeps matching.

### 2. A full GPT-2-style decoder layer composed through the on-device cache — DONE

`IlgpuEngine.DecodeLayerStep(GpuKvCache cache, Tensor<float> hidden, DecoderLayerWeights w)` runs ONE whole
pre-LayerNorm decoder layer entirely on the GPU, threaded through the persistent cache. It chains, all on
device, reusing the existing kernels:

1. `ln1 = LayerNorm(hidden)` (LayerNorm kernel);
2. fused QKV projection `ln1·Wqkv + bqkv` (MatMul + broadcast-Add) → split columns into per-head Q/K/V
   `[H,S,D]` (Gather kernel with host-precomputed offsets);
3. append K/V into the persistent `GpuKvCache` (device→device `SubView.CopyTo`, no realloc/host round-trip)
   and scaled-dot-product attention over the **entire cached prefix** per head
   (Transpose → MatMul → scale → Softmax → MatMul) — K/V stay on-device across steps;
4. output projection `ctx·Wo + bo` and the **attention residual** add;
5. `ln2 = LayerNorm(h2)`, the two-matmul **GELU MLP** (`fc·Wfc+b` → Gelu kernel → `·Wproj+b`), and the
   **MLP residual** add.

Only the final hidden state `[S,E]` is downloaded. `DecoderLayerWeights` carries the conventional HF/GPT-2
Conv1D `[in,out]` weight layout. Validated by `GpuDecoderLayerTests.Cuda_DecodeLayerStep_MultiStep_Matches_Cpu`
against a hand-written CPU reference (using the engine's own A&S-Erf GELU) over 4 decode steps to 1e-3; step 0
exercises the seq-0 empty-cache path.

### 3. Whole-graph distilgpt2 GPU decode e2e — NOW REQUIRED, NO FALLBACK

The full distilgpt2 ONNX graph (1569 nodes) now executes **end-to-end through `IlgpuEngine.Run` with no
fallback**, matching `ManagedCpuEngine` logits/argmax. Two op handlers were generalized to route the int/bool
shape subgraph host-side (the previous blocker was an op handler calling `FloatBuf` on a host-side Int64
tensor):

- **`Gather`** — added an integer/bool *data* path. Gathering elements out of a host int tensor (e.g. selecting
  dims from a `Shape` vector, or token ids) now stays host-side via `HostGatherFlat`, mirroring the float
  device-gather offset math; only float `data` touches the device. (Previously `Gather` assumed float `data`
  and threw on `/transformer/Shape_output_0` — the exact reported blocker.)
- **`Add`/`Sub`/`Mul`/`Div`** (`RunBinary`) — added a host-side integer path (`RunBinaryHost`/`BinaryHost<T>`)
  for shape/index arithmetic where both operands are int64/int32. It mirrors the CPU `BroadcastBinaryKernel`
  exactly (NumPy broadcasting, **integer truncating division**) and preserves the int dtype so a downstream
  Gather/Slice/Reshape still receives integers. Float operands keep the on-device fast/broadcast path unchanged.

Every other op in distilgpt2's prologue was *already* dtype-routed correctly (Reshape/Unsqueeze/Squeeze via
`Reshaped`, Concat/Slice/Expand/Split/Where with explicit host int branches, Cast bridging int-host↔float-device,
and the six integer/mask prologue ops Range/ConstantOfShape/Equal/Greater/Trilu/ScatterND host-side). With the
two fixes above the **whole graph clears**.

`GpuWholeGraphTests.Cuda_DistilGpt2_WholeGraph_Logits_Match_Cpu` is asset-gated (discovers `distilgpt2.onnx`
via `MODELSHARP_MODELS_DIR` → repo-relative `models/` → `/home/x16/models`) and CUDA-gated; it skips cleanly
when either is absent. When the model **is** present the whole-graph `Run` is **REQUIRED** — any exception is a
hard failure (the report-and-skip fallback was removed) and so is a numeric mismatch. It runs the full ONNX
graph through `IlgpuEngine.Run` with empty (seq-0) `past_key_values.*` float feeds and asserts last-position
**logits parity** (max|Δ| < 5e-2; observed **1.8e-4** on an RTX 4090) and an exact **greedy argmax** match vs
`ManagedCpuEngine` for two decode steps (step 0 argmax 274, step 1 argmax 389 — both agree).

`GpuWholeGraphTests.DistilGpt2_WholeGraph_CpuRouting_Matches_Cpu` is the CUDA-independent companion: it drives
the same whole graph through `IlgpuEngine.Run` on ILGPU's **CPU** accelerator (`preferCpu: true`). The int/float
routing is device-agnostic, so this proves the whole graph clears end-to-end and matches `ManagedCpuEngine`
**bit-for-bit (max|Δ| = 0.0)** even on machines with no GPU.

## Honest list of what still falls back / is not covered

- The 6 integer/mask prologue ops (`Range`/`ConstantOfShape`/`Equal`/`Greater`/`Trilu`/`ScatterND`) are
  dispatched **host-side** — they run on the CPU-resident int/bool tensors (no GPU kernel), consistent with
  the engine's design where int/bool tensors always live on the host. This keeps the whole graph
  GPU-dispatchable without a CPU fallback; the float compute path is unaffected.
- `DecodeStepAttention` (the original bare self-attention seam) is retained but **superseded** by
  `DecodeLayerStep` for the full-layer path. It downloads the per-step context to host at the end of each
  step; `DecodeLayerStep` instead keeps the whole layer on-device and downloads only the final hidden state.
- `DecodeLayerStep` covers a single decoder layer; a multi-layer stack is the caller's loop (feed layer *i*'s
  output hidden state as layer *i+1*'s input, one `GpuKvCache` per layer). It assumes the GPT-2 pre-LN layout
  (fused `c_attn` QKV, MLP = `c_fc`→GELU→`c_proj`) and a single batch row of `[stepLen, embed]`. This stateful
  cache seam is now an *alternative* to — not a prerequisite for — running the whole graph: the stateless
  `IlgpuEngine.Run` drives the entire distilgpt2 ONNX graph end-to-end on its own (see #3 above).
- **Nothing in distilgpt2 falls back any longer.** The whole graph executes end-to-end through the stateless
  `IlgpuEngine.Run` on real CUDA hardware, matching `ManagedCpuEngine` (verified — see #3). The only host-side
  computation is the integer/bool shape & mask subgraph (Shape/Range/ConstantOfShape/Equal/Greater/Trilu/
  ScatterND, plus int Gather/Concat/Slice/Reshape/Unsqueeze/Squeeze/Expand/Split/Where/Cast and int
  Add/Sub/Mul/Div index math). That is **by design**, not a fallback: int/bool tensors always live host-side in
  this engine (ILGPU's portable backends have no native int64, and these tensors feed pure index math), the
  float matmul/attention/MLP path runs entirely on the device, and `Cast` is the only host↔device seam. There is
  no CPU-engine fallback and no unsupported op.

## Per-op CPU fallback — the GPU engine now runs any CPU-runnable model

The GPU engine is no longer limited to the ops it has native kernels for. `IlgpuEngine.Run`'s `switch
(node.OpType)` previously ended in `default: throw UnsupportedOperatorException`; that `default` now routes the
node through the managed CPU kernel registry (`RunCpuFallback`). The result: **any graph the
`ManagedCpuEngine` can run, the GPU engine can run too** — accelerated where a native GPU kernel exists,
correct via the CPU fallback everywhere else. The native distilgpt2 path is unchanged (it never hits the
fallback), so there is no regression on the fully-native graphs.

### How the fallback crosses the device↔host boundary

`RunCpuFallback(node, values)` mirrors exactly how `ManagedCpuEngine.ExecuteNodes` invokes a kernel for one
node:

1. **Inputs → host.** Each input `DeviceValue` is materialized to a dtype-carrying host `Tensor` via
   `DeviceValue.ToHost()`: float buffers are **downloaded** from the device; int/bool tensors already live
   host-side and pass through untouched. Omitted/optional inputs (empty name) are skipped, exactly as the CPU
   engine leaves them unbound.
2. **Run the CPU kernel.** Those inputs seed a `GraphContext` (the same `Dictionary<string,Tensor>` environment
   the CPU engine threads through kernels); the kernel for `node.OpType` is looked up in a **lazily-built**
   `KernelRegistry.CreateDefault()` (the same registry `ManagedCpuEngine` uses) and `kernel.Execute(node, ctx)`
   writes its named outputs into the environment — identical call shape to the CPU engine.
3. **Outputs → DeviceValue.** Each declared output tensor is re-homed via `Load(...)`: **float outputs are
   re-uploaded to the device** (so the next GPU op reads a device buffer, not a host array), while int/bool
   outputs are kept host-side — dtype is preserved throughout. The re-uploaded buffers are tracked in the same
   `values` map and disposed by `Run`'s existing dedup-by-reference cleanup.

The registry is built **lazily** (first fallback only), so fully-native graphs (distilgpt2) never construct
it and pay nothing. A native GPU op composes seamlessly with a fallback op in either direction because both
sides speak `DeviceValue` — e.g. `MatMul (native GPU) → Softplus (CPU fallback) → Add (native GPU)` keeps the
right buffers on the device at each hop (`GpuFallbackTests`).

### "Unsupported" now means no engine supports it

If the CPU registry *also* lacks a kernel for the op, `RunCpuFallback` throws `UnsupportedOperatorException`
with a message stating that neither the GPU engine nor the CPU kernel registry can run it. So the exception's
meaning has sharpened from "the GPU engine doesn't have this kernel" to "**no engine** supports this op."

Control-flow ops (`If`/`Loop`/`Scan`) that need nested-subgraph execution are dispatched to their CPU kernels
too, but the fallback installs a bare `GraphContext` with no subgraph runner; such a kernel surfaces a clear
error from `GraphContext.RunSubgraph` rather than miscomputing. Running those through the GPU engine's stateless
`Run` is out of scope here (they require the engine-level subgraph runner the CPU engine wires up).

### A few extra native GPU kernels

Four trivially-native unary float ops were added so a fallback wouldn't needlessly download for them: `Log`
(`MathF.Log`/`LogF` intrinsic), `Abs`, `Neg`, and `Reciprocal`. They are guarded with
`when values[node.Inputs[0]].IsFloat`, so integer inputs (rare — e.g. int `Abs`/`Neg`) still fall through to
the dtype-correct CPU kernel.

### Tests (`GpuFallbackTests.cs`, CUDA-gated, `[Collection("CudaGpu")]`)

- **Single fallback op round-trips:** `Softplus`, `ReduceMax` (a reduction the GPU switch lacks), and `Tile`
  (a data-movement op with an int `repeats` input) each match `ManagedCpuEngine` to 1e-4 through `IlgpuEngine`,
  proving the device↔host round-trip is correct.
- **No engine supports it → throws:** a made-up op type throws `UnsupportedOperatorException` (runs on ILGPU's
  CPU accelerator, so it needs no CUDA).
- **Mixed native + fallback end-to-end:** `MatMul → Softplus → Add` and a chained
  `MatMul → Mish → ReduceMax → Add` both match the CPU engine, proving fallback outputs re-enter the device for
  the trailing native ops.
- **No native regression:** a synthetic distilgpt2-style graph (`MatMul/Add/Gelu/LayerNormalization/Softmax`)
  matches the CPU engine on both the CPU accelerator and real hardware, confirming the native paths are
  untouched.

---

# Quantized LLM inference on the GPU engine (this pass)

## What runs

Quantized inference runs **end-to-end through `IlgpuEngine`**, and the GEMM hot-spot — `MatMulInteger` — now
runs **natively on the device** (no longer through the CPU fallback). The rest of the quantized op family
(`DequantizeLinear`/`QuantizeLinear`/`DynamicQuantizeLinear` and the `QLinear*` family) still routes through the
**per-op CPU fallback** (`RunCpuFallback`), which is cheap (per-tensor/per-channel scalar math, not the O(MNK)
GEMM). This composes correctly with the surrounding **native GPU float ops**: in a real quantized decoder the
heavy float matmuls (attention QKᵀ / A·V, and the `lm_head` logits projection over the full vocab) dispatch to
the device `MatMul` kernel, the INT8 linear layers now dispatch to the device `MatMulIntegerK` kernel, and only
the lightweight quant/dequant glue rounds through the host.

## Native `MatMulInteger` GEMM (this pass) — on-device, no longer a fallback

`MatMulInteger` previously had **no native case** in the engine switch, so it hit `RunCpuFallback` → the scalar,
**double-precision** `MatMulIntegerKernel` on the host (with a device↔host round-trip per linear). It is now a
first-class native GPU op:

- **Kernel** (`GpuKernels.MatMulIntegerK`): one thread per output element across `[batch, M, N]`, exactly the
  layout of the float `MatMulK`. A and B are uploaded to the device as **int32** buffers (the uint8/int8 operands
  are widened to int32 host-side in `ReadQuantAsInts`). Each thread computes `Σ_k (a−a_zp)·(b−b_zp)` with an
  **int32 accumulator**. The per-flattened-batch operand base offsets (NumPy-broadcast-resolved, element units)
  are precomputed on the host and uploaded, identical to `RunMatMul`, so batching/broadcasting of the leading
  dims matches the float path and the CPU reference.
- **Zero-point handling**: zero points are uploaded as int32 buffers with a per-operand **stride selector** —
  `aZpStride`/`bZpStride` ∈ {0,1}. Stride 0 means *absent or per-tensor* (every thread reads the single scalar
  `zp[0]`); stride 1 means *per-row of A* (`zp[m]`, length M) or *per-column of B* (`zp[n]`, length N). This one
  branchless form covers all four cases (absent / per-tensor / per-row / per-column) and reproduces the CPU
  kernel's selection exactly — including the `M==1`/`N==1` edge where a length-1 zp is treated as scalar.
- **Dtype variants**: uint8×int8 (the gpt2 case), uint8×uint8, int8×int8, int8×uint8 — and int32 — all work,
  because operands and zero points are widened to int32 before upload (`ReadQuantAsInts`).
- **Exactness**: the CPU kernel accumulates each product in int64 then casts `(int)sum`; the GPU kernel
  accumulates in int32 (wrapping). Two's-complement addition has identical low-32 bits at any width, and every
  `(a−zp)` difference and its product fit in int32 (operands ∈ [−128,255]), so the int32 result is **bit-exact**
  vs the CPU reference — proven element-exact by the parity tests below.
- **Result flow downstream**: the int32 device result is read back into a host `Tensor<int>` and stored as a
  host-side `DeviceValue` — the **same dtype and home** the CPU fallback produced. So the downstream
  `Cast(int32→float)` → `Mul(scale)` dequant chain (and any `DequantizeLinear` consumer) sees an identical value
  and is **unchanged**. (Int32 tensors live host-side in this engine by the same int/bool convention as token
  ids and masks; only the O(MNK) compute moved to the device, which is the point.)

Wired as `case "MatMulInteger": RunMatMulInteger(...)` in `Run`'s switch.

## Validation

### Synthetic quantized-graph parity (`tests/ModelSharp.Tests/GpuQuantizedTests.cs`) — must-pass, no asset

Small in-memory `ModelGraph`s exercise the quantized path and assert `IlgpuEngine` == `ManagedCpuEngine`
(integer/byte outputs element-exact; float outputs to 1e-4). Each runs on ILGPU's **CPU accelerator** as a
plain `[Fact]` (covered on every machine, no CUDA) and is re-run on **hardware CUDA** in the nested
`Cuda` class (`[Collection("CudaGpu")]`, asserts `IsHardwareGpu`, skips cleanly with no CUDA):

| Graph | What it proves |
|-------|----------------|
| `MatMulInteger` → `Cast(int32→float)` → `Mul(scale)` | INT8 matmul with scalar zero-points, dequantized; the **native** int32 GEMM output bridges to a device float via `Cast` |
| bare `MatMulInteger` (`MatMulInteger_Native_Parity_CpuAccel`, `[Theory]` over 12 cases) | the native int32 GEMM is **element-exact** vs the CPU reference across shapes (incl. M==1, N==1, larger tiles), operand dtypes (u8×i8, u8×u8, i8×i8, i8×u8) and zero-point forms (absent / per-tensor / per-row A / per-col B / batched rank-3 & rank-4). Re-run on CUDA via `Cuda_MatMulInteger_Native`. |
| `QuantizeLinear` → `QLinearMatMul` → `DequantizeLinear` | full per-tensor quantize→qmatmul→dequant round trip |
| per-channel `DequantizeLinear`(axis=0) → `Transpose` → `MatMul` → `Add` | INT8 **per-channel** weight dequant feeding a native GPU matmul+bias |
| `DynamicQuantizeLinear`→`MatMulInteger`→dequant → softmax self-attention → `DynamicQuantizeLinear`→`MatMulInteger`→dequant | tiny **quantized transformer block**: quantized fallback ops interleaved with native GPU `MatMul`/`Transpose`/`Softmax`/`Mul` end-to-end |
| `QLinearAdd` → `QLinearMul` | quantized elementwise (residual-style) with per-tensor scales |

### Real quantized LLM (`tests/ModelSharp.Tests/RealModels/QuantizedGpt2GpuTests.cs`) — asset-gated

A genuinely-quantized ONNX LLM was downloaded: **`onnx-community/gpt2-ONNX` `model_quantized.onnx`**
(ONNXRuntime dynamic-INT8 quantization of gpt2, **267 MB on disk**, a `text-generation-with-past` export).
Its quantized linears lower to **48× `DynamicQuantizeLinear` → `MatMulInteger`** plus `DequantizeLinear`,
interleaved with **25 plain float `MatMul`s** (24 attention + the `lm_head` `[seq,768]@[768,50257]` logits
projection), `Softmax`, LayerNorm-via-primitives (`ReduceMean`/`Sub`/`Pow`/`Sqrt`), Tanh-GELU, and a 12-layer
with-past KV-cache (`input_ids`/`attention_mask`/`position_ids` + empty-`past` prefill — same contract as the
already-GPU-validated distilgpt2, plus `position_ids`). Every op it uses is in the CPU registry (native GPU or
fallback), so the whole graph is GPU-dispatchable.

The test drives the **whole quantized graph** through `IlgpuEngine.Run` for several **greedy decode steps**
(stateless re-prefill with the growing token sequence) and asserts the last-token logits match
`ManagedCpuEngine` (max|Δ| < 5e-2) and that **GPU argmax == CPU argmax at every step**, so the decoded id
sequence is coherent and deterministic on the GPU engine. A CPU-accelerator routing `[Fact]` runs the full
INT8 graph everywhere; the `Cuda_*` test re-runs it on the RTX 4090. Discovery is via
`MODELSHARP_MODELS_DIR` → repo `models/` → `/home/x16/models/gpt2-quantized.onnx`, skipping cleanly if absent
(never hard-fails on a missing download). The asset is gitignored (267 MB; not committed).

(An ONNXRuntime reference greedy-decoded `"The quick brown fox"` → ids `[274, 389, 257, 1310]` deterministically,
confirming the model itself is sane; the ModelSharp test asserts the engine-internal GPU-vs-CPU argmax
invariant rather than exact token values, which is robust to small absolute-logit differences.)

## Scale characterization

**Memory footprint per quantization** (weights only; activations/KV add to this at runtime):

| Quantization | bytes/param | gpt2 (124 M params) on disk | 7B-class model (weights) |
|--------------|-------------|------------------------------|---------------------------|
| fp32         | 4.0         | ~498 MB (`model.onnx`)       | ~28 GB                    |
| fp16         | 2.0         | ~249 MB (`model_fp16.onnx`)  | ~14 GB                    |
| **int8**     | **1.0**     | **~267 MB (`model_quantized.onnx`, measured)** | **~7 GB**       |
| int4 / uint4 | 0.5 (+scales)| ~249 MB (`model_q4f16.onnx`)| **~3.5–4 GB**             |

So an INT8 7B fits in the RTX 4090's 24 GB VRAM with wide headroom for activations + KV cache; INT4 roughly
halves that again. The KV cache (fp16) for a 7B model (≈32 layers × 2 × n_kv_head·head_dim) is ~0.5 MB/token,
so even a 4k-token context (~2 GB) stays comfortably within 24 GB.

**Throughput observation.** The INT8 GEMM hot-spot is now **on-device**. `GpuMatMulIntegerPerfTests.cs`
(CUDA-gated, `[Collection("CudaGpu")]`) times the native `MatMulIntegerK` kernel against the same op on the
managed CPU engine (the old double-precision `MatMulIntegerKernel` fallback) on two realistic shapes and asserts
the native int32 result is **element-exact** vs CPU before logging the speedup (no ratio is hard-asserted —
hardware-dependent):

- `Square2048` — a 2048×2048×2048 INT8 GEMM (~17.2 GOP).
- `Llm[32,4096]×[4096,4096]` — an LLM-decode-shaped projection (seq=32, hidden=4096), the shape a 7B-class
  attention/MLP linear lowers to at INT8.

*Expected* magnitude on a 4090: the CPU reference is a single-threaded scalar `double` triple-loop, so on these
sizes the native per-output-element GPU kernel should land in the **~50–200×** range for the 2048³ point and
**tens of ×** for the LLM-shaped point (the device kernel is one-thread-per-output naïve, not yet shared-memory
tiled, so it is occupancy/issue-bound rather than peak-INT8-DP4A bound — there is headroom). The exact numbers
print to the test output on the run box; the parent's central run captures them. Either way the per-linear
device↔host round-trip and the host double-precision GEMM are **gone** for `MatMulInteger`.

**What now falls back vs runs native (quantized path):**

- **`MatMulInteger` — NATIVE on device** (this pass). The O(MNK) GEMM, the actual hot-spot.
- `DynamicQuantizeLinear` / `QuantizeLinear` / `DequantizeLinear` / `QLinear*` — still CPU fallback, but these
  are O(N) per-tensor / per-channel scalar passes (quantize an activation, rescale an int32 accumulator), not
  GEMMs. They remain candidates for a native pass but are not the throughput bottleneck.
- All float ops (attention QKᵀ/A·V, `lm_head`, LayerNorm-via-primitives, GELU, softmax) — native on device.

**What a literal 7B-class quantized run needs** (exact remaining requirements):

1. **Asset.** A 7B INT8/INT4 ONNX export with a tokenizer (e.g. an `onnx-community`/`optimum`-quantized
   Llama/Mistral-7B `*-with-past` export, ~7 GB INT8 / ~4 GB INT4). Drop it in `MODELSHARP_MODELS_DIR`; the
   asset-gated test pattern picks it up. (Not downloaded here — out of test-time budget; the path is proven on
   the smaller real quantized gpt2, which shares the identical op structure.)
2. **VRAM.** ~7 GB (INT8) / ~4 GB (INT4) weights + ~1–2 GB activations + KV cache — fits the 24 GB 4090.
3. **GEMM throughput — NOW NATIVE.** The native on-device `MatMulInteger` (uint8/int8 → int32 accumulate, fed the
   host-resident quantized tensors directly, computed entirely on the accelerator) removes the per-linear
   double-precision host GEMM and its round-trip — the one piece of real engineering that previously stood
   between *possible* and *performant*. The remaining throughput upside is incremental, not blocking: a
   shared-memory-tiled / DP4A INT8 GEMM (vs the current one-thread-per-output kernel) and native
   `DynamicQuantize`/`Dequantize` to eliminate the last lightweight round-trips. With the native GEMM in place, a
   7B INT8 decode on a 4090 is memory-bandwidth bound (weights ~7 GB/token-pass), i.e. tens of tokens/s is the
   expected regime once the asset is present.

---

## B-GPU-2 — more native kernels + shared-memory-tiled int8 GEMM

This pass (a) moves a batch of high-frequency ops off the per-op CPU fallback onto native device kernels, and
(b) replaces the naïve one-thread-per-output `MatMulInteger` GEMM with a shared-memory-tiled kernel.

### Tiled int8 GEMM (`GpuKernels.MatMulIntegerTiledK`)

- **Design.** A 2-D grouped launch of `IntTile×IntTile` (16×16 = 256) threads computes one `M×N` output tile per
  group. Each K-step stages an `IntTile×IntTile` block of A′ and B′ into flat 1-D shared memory (one element per
  thread, coalesced), `Group.Barrier()`s, then accumulates the tile's partial dot product; a second barrier
  guards the next stage. Edge tiles (M/N/K not a multiple of 16) read zeros into shared memory and skip the
  out-of-range store. One grouped launch per broadcast-resolved batch (serialized on the default stream).
- **Zero points are pre-subtracted on the host** into int32 `A′ = A − a_zp`, `B′ = B − b_zp` (covering absent /
  per-tensor / per-row-A / per-column-B). The kernel then does plain int32 multiply-accumulate, **bit-identical**
  to the naïve kernel's `(A−az)·(B−bz)` int32 fold — and to the CPU `MatMulIntegerKernel`'s `(int)` cast of its
  int64 accumulator for in-range models. `int32` wraparound semantics preserved.
- **DP4A.** ILGPU 1.5.3 exposes no portable `__dp4a` / 4×int8-dot intrinsic that JIT-lowers across the
  CUDA *and* CPU backends, so the kernel stays an **int32-accumulate tiled** GEMM (correct + portable, tested on
  the CPU accelerator on every machine). DP4A would be a CUDA-only follow-up behind a backend probe; the tiling
  is the portable, test-covered win.
- **Benchmark.** `GpuMatMulIntegerPerfTests` times tiled vs naïve vs CPU-ref on `2048³` and an LLM-decode-shaped
  `[32,4096]×[4096,4096]`, asserts both GPU kernels are element-exact vs CPU first, then logs the
  `tiled-vs-naive` and `tiled-vs-CPUref` speedups (no ratio hard-asserted — hardware-dependent). The naïve kernel
  is kept and reachable via the `IlgpuEngine.UseNaiveIntGemm` test seam for the head-to-head. OOM-skips cleanly.

### New native float kernels (were CPU-fallback)

- **Extra unary / activations** (one thread per element, each mirrors its CPU kernel exactly):
  `Sign`, `Floor`, `Ceil`, `Round` (banker's), `Softplus`, `Mish`, `HardSwish`, `HardSigmoid` (α,β),
  `Elu` (α), `Selu` (α,γ), `Clip` (min/max from inputs or attrs). Integer inputs (rare; e.g. int `Sign`) still
  fall through to the CPU kernel so dtype stays correct.
- **`ReduceMax` / `ReduceMin` / `ReduceProd`** — an op-selected device reduce (`GpuKernels.ReduceOpK`) sharing the
  exact host-side axis-resolution / offset-precompute and row-major fold order of the existing Sum/Mean reduce.
- **Variadic `Min` / `Max` / `Sum` / `Mean`** — an accumulator buffer seeded with the op identity, each input
  folded in via `GpuKernels.VariadicFoldK` (NumPy broadcasting), Mean's divide a final scalar-broadcast pass.
  Only when every input is a device float (matches the CPU kernel's float-only path); otherwise CPU fallback.
- **`Pad`** (constant / edge / reflect, float) and **`Tile`** (float) — a host-precomputed per-output-element
  source offset gathered on-device via the existing Gather kernel; Pad's constant fill written into the padded
  positions on-device with a scalar-broadcast `Where`. Integer/bool Pad/Tile stay on the CPU fallback (dtype).
- **`Less`** added to the host-side `Equal`/`Greater` comparison path (bool outputs are host-resident in this
  engine by design — never on the float compute path — so comparisons compute host-side, no device kernel).

### What still falls back, by design

- Quantized glue (`DynamicQuantize`/`Quantize`/`Dequantize`/`QLinear*`) — O(N) scalar passes, not GEMMs.
- The long tail of low-frequency ops (RNN/signal/sequence/control-flow), int/bool elementwise & shape math, and
  the integer/bool variants of the ops above — all correct via the per-op CPU fallback.

### Tests

- Parity for every new native kernel vs `ManagedCpuEngine`: CPU-accelerator `[Fact]`/`[Theory]` (run on any
  machine) in `GpuOpsExtraTests`, plus hardware-CUDA re-runs in `GpuNativeOpsCudaTests` (assert `IsHardwareGpu`,
  skip green with no CUDA, **skip on device `out of memory`**).
- int8 GEMM bit-exactness: `GpuQuantizedTests.MatMulInteger_TiledVsNaive_BitExact_CpuAccel` asserts tiled == naïve
  == CPU across all shape/dtype/zp cases; `GpuMatMulIntegerPerfTests` re-asserts on CUDA and logs the speedup.

---

# Native on-device `MatMulNBits` (INT4/INT8 block-quant) + `GroupQueryAttention` (this pass)

The two heavyweight quantized-LLM contrib ops that the onnxruntime-genai INT4 LLM exports (Qwen-0.5B,
Mistral-7B, Phi-3, Llama) lean on — `MatMulNBits` (block-wise INT4/INT8 weight-quantized linear) and
`GroupQueryAttention` (packed-QKV grouped-query attention with in-op rotary + KV-cache) — previously had **no
native case** in the engine switch, so each hit `RunCpuFallback`: download the float inputs, run the scalar CPU
kernel, re-upload. For a 7B INT4 model those two ops *are* essentially the whole decoder, so the fallback
round-trip dominated runtime. Both are now **first-class native GPU ops**, with the float compute path staying
on the device and no per-op host round-trip.

## `MatMulNBits` — dequant-in-kernel block-quant GEMM

- **Kernel** (`GpuKernels.MatMulNBitsK`, one thread per output element across the flattened `[M, N]`). A (float
  `[..., K]`) lives on the device; the packed quantized B (`[N, nBlocksPerRow, blobSize]` uint8, widened one int
  per byte), the per-`(row, block)` `scales`, and the zero points are uploaded. Each thread walks the K-blocks of
  its weight row, **unpacks each n-bit code and dequantizes `W[n,k] = (q − zp) · scale` inside the kernel**, and
  accumulates `Σ_k A[m,k]·W[n,k]` in float — writing the float `[..., N]` result directly. No host dequant scratch.
- **Bit-for-bit dequant semantics** vs the CPU `MatMulNBitsKernel`: least-significant-first nibble packing (bits=4:
  even index = low nibble, odd = high nibble; bits=8: one byte/value), default symmetric zero point `2^(bits-1)`
  when absent, partial last block (`K % blockSize`), and the same per-block accumulation order. `g_idx` with a
  non-trivial block permutation throws (matches CPU).
- **Zero-point forms** (branchless via a `zpMode` selector in the `MatMulNBitsParams` struct): `0` = default
  symmetric, `1` = float per-`(row, block)` (`zpFloat[n·nBlocksPerRow+b]`), `2` = packed n-bit (same nibble
  packing as B, row stride `zpRowBytes`). Unused zp buffers are 1-element sentinels.
- **Arity note:** ILGPU 1.5.3's `LoadAutoGroupedStreamKernel<…>` caps at **15 generic params (incl. `Index1D`)**.
  The 9 scalar layout params are bundled into a blittable `MatMulNBitsParams` struct (like `ConvParams`) so the
  delegate is `Index1D + 6 views + 1 struct`. (`GqaAttentionK` lands at exactly 15 and is fine as-is.)
- Wired as `case "MatMulNBits": RunMatMulNBits(...)`. The float result is a normal device `DeviceValue`, so the
  downstream Add/residual/LayerNorm chain consumes it on-device with no round-trip.

## `GroupQueryAttention` — attention on the device

`RunGroupQueryAttention` does the whole op on the accelerator, composing three new kernels with device→device copies:

1. **Packed-QKV split** (genai layout: K/V input names empty, `query` holds Q|K|V concat on the last dim) — split
   per `(b, s)` row into device Q/K/V buffers `[B, S, heads·head_dim]` via `SubView.CopyTo` (no host round-trip).
   The unpacked layout copies the separate K/V into fresh mutable buffers (rotary mutates Q/K in place).
2. **In-op rotary** (`do_rotary=1`): `GpuKernels.RotaryK`, one thread per `(b, s, head, j)` rotary pair, applies
   RoPE to Q and K (not V) using the `cos`/`sin` caches indexed by absolute position `pastSeq+s`. Both NeoX
   half-split (`rotary_interleaved=0`) and GPT-J interleaved (`=1`) conventions, mirroring the CPU `ApplyRotary`.
3. **present-K/V build**: past `[B, kvh, pastSeq, hd]` staged into present `[B, kvh, totalSeq, hd]` by per-head
   `SubView.CopyTo`, then the new K/V scattered in at offset `pastSeq` by `GpuKernels.GqaScatterKvK`.
4. **Attention core** (`GpuKernels.GqaAttentionK`, one thread per `(b, h, qi)` output row): repeat-KV grouping
   (`g = h / groupSize`), **causal** bound `min(pastSeq+qi, seqlens_k[b])`, an online two-pass softmax
   (max → Σexp → V-weighted sum) over the cached prefix — matching the CPU GQA accumulation order. The `output`
   and (when named) `present_key`/`present_value` are device outputs.

`scale` defaults to `1/√head_dim`; `seqlens_k`/`total_sequence_length` tolerated; `local_window_size` not
implemented (treated as disabled), same as the CPU kernel. Wired as `case "GroupQueryAttention": RunGroupQueryAttention(...)`.

## Do the Qwen/Mistral graphs now run these ops native?

Yes for the two target ops: a genai INT4 graph's INT4 linears (`MatMulNBits`) and every attention block
(`GroupQueryAttention`) now dispatch to the device kernels above — the float compute path is resident on the
accelerator with **no per-op host round-trip** for them. The remaining genai glue (`RotaryEmbedding` outside GQA
if any, `SkipSimplifiedLayerNormalization`, etc.) still uses native float kernels where they exist and the per-op
CPU fallback otherwise (lightweight, not the GEMM/attention hot-spots). A full end-to-end Qwen/Mistral run is
asset-gated (the 7B export is not committed); the op-level native path and CPU-parity are proven by the tests below.

## Tests (`tests/ModelSharp.Tests/GpuQuantizedLlmOpsTests.cs`)

Parity vs `ManagedCpuEngine` to a **relative** tolerance (`1e-3·max(1,|cpu|) + 1e-4` absolute floor — GPU float
accumulation differs slightly from the CPU reference, so relative, not bit-exact). Each case is a CPU-accelerator
`[Theory]` (runs on **every machine, no CUDA**) plus a CUDA re-run in the nested `Cuda` class
(`[Collection("CudaGpu")]`, asserts `IsHardwareGpu`, skips cleanly on no-CUDA **and** on device `out of memory`).

- `MatMulNBits_Native_Parity_CpuAccel` (10 cases): bits 4 & 8, block sizes 4/8/16/32, partial last block, all three
  zero-point forms (default / float per-block / packed n-bit), various M×K×N incl. odd N.
- `Gqa_Packed_Parity_CpuAccel` (5 cases): packed-QKV with/without rotary, repeat-KV group sizes (incl. MHA where
  kv==q heads), batch>1 — asserts `output` + `present_key` + `present_value`.
- `Gqa_DecodeWithPast_Parity_CpuAccel` (3 cases): unpacked single-token decode over a non-empty past with
  `seqlens_k`, asserting the context and the appended present-K/V.

Verified locally on the ILGPU **CPU accelerator** (standalone, outside the shared project build): native GPU vs
`ManagedCpuEngine` `maxΔ = 0` across MatMulNBits (INT4 default-zp, INT8 packed-zp), GQA packed+rotary
(`output`/`present_key`/`present_value`), and GQA decode-with-past — all well within tolerance.

## Note for the central build

- New **public** type `ModelSharp.Gpu.MatMulNBitsParams` (blittable struct, mirrors `ConvParams`) — additive,
  no existing API changed.
- Edits confined to the allowed file-set: `IlgpuEngine.cs`, `GpuKernels.cs`, `B5_NOTES.md`, and the **new**
  `GpuQuantizedLlmOpsTests.cs`. No `.csproj`, core, CPU, or generation files touched.
- Per the boundary, `dotnet build`/`dotnet test` were **not** run (shared bin/obj). Compilation was verified by a
  standalone `csc` build of the three Gpu sources (exit 0) and of the new test file (exit 0); functional parity
  was verified by a standalone run on the ILGPU CPU accelerator. The parent should run the full suite to
  integrate.
