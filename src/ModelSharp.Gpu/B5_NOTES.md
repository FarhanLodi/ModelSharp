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

## distilgpt2 GPU-op audit (see `GpuDistilGpt2AuditTests` and `GpuLlmTests.DistilGpt2_Gpu_Coverage_*`)

- **1548 / 1569 nodes (98.7%) are GPU-dispatchable** after this pass.
- **Still falls back to CPU** (21 nodes, 6 distinct ops), all in the mask / position-id prologue:
  `Range`, `ConstantOfShape`, `Equal`, `Greater`, `Trilu`, `ScatterND`.
  These are integer/boolean control-flow ops that build the causal mask and position ids; they are not
  on the float compute path. The first one in topo order is `/transformer/Range` (node #23).
- Consequently a **whole-graph** distilgpt2 GPU run is **not** reachable yet (it would stop at `Range`),
  and is also impractical to drive through `Run` because the empty (seq-0) `past_key_values` float
  inputs produce zero-length device allocations. The transformer **compute** path itself
  (Gemm/MatMul/Softmax/LayerNorm/attention, the Pow+Tanh GELU, and all the Reshape/Transpose/Split/
  Concat/Gather/Slice plumbing) is GPU-complete.

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

## Honest list of what still falls back / is not covered

- The 6 integer/mask prologue ops above (`Range`/`ConstantOfShape`/`Equal`/`Greater`/`Trilu`/
  `ScatterND`) are CPU-only; no GPU kernels were added for them this pass.
- `DecodeStepAttention` is a single self-attention block, not the whole decoder layer stack (no
  projections/MLP/residual wired into the seam) — those ops exist individually on the GPU but are not
  composed into the cache path here.
- The cache currently downloads the per-step context to host at the end of each step (the typical
  consumer wants the token's hidden state on host for the next op); the K/V themselves never leave the
  device.
