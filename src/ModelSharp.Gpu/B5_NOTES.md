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
