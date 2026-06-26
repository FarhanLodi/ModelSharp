# B5 — Whole-graph on GPU & on-device KV-cache: current state

## Current data-movement behaviour (after this change)

`IlgpuEngine.Run` already keeps **all intermediate tensors resident on the device** across the
op chain. The mechanics:

- `buffers` is a `Dictionary<string, MemoryBuffer1D<float>>` of **device** buffers. Inputs and
  initializers are uploaded **once** at the start of `Run`. Every op reads its input device
  buffers and writes a freshly-allocated device output buffer into `buffers`; nothing is copied
  back to host between ops.
- There is a **single** `_accelerator.Synchronize()` at the very end, immediately before the
  outputs are read back with `GetAsArray1D()`. So a full forward pass issues all kernels
  asynchronously and pays exactly **one** device→host copy per graph output — not per op.

In other words, the per-op host round-trip that B5 warns about **was already absent** for the
main op chain. The kernels (Add/Sub/Mul/Div, activations, Transpose, Softmax, Reduce, MatMul,
Conv) all operate device→device.

### Host round-trips that remained, and what was done

1. **`ReduceSum`/`ReduceMean` identity path** (`noop_with_empty_axes` with empty axes). Previously
   did `Allocate1D(x.GetAsArray1D())` — a device→host→device bounce. **Fixed**: now a
   device-to-device `x.View.CopyTo(stream, copy.View)`, no host involvement.

2. **`ReduceSum`/`ReduceMean` dynamic axes** read from a *tensor* input
   (`buffers[node.Inputs[1]].GetAsArray1D()`). This is an unavoidable host read: the axes are
   needed on the **host** to compute the reduction's stride layout before launching the kernel.
   It only triggers when axes are supplied as a runtime tensor rather than an attribute (rare for
   static graphs), and moves a handful of ints, not bulk data. Left as-is.

The stride/offset/batch-offset arrays the host precomputes for broadcasting, Transpose, Reduce and
MatMul are uploaded as small `int[]` device buffers (`Upload`) — these are control data derived
from shapes, not tensor payload, and are correctly kept off the hot path.

## What remains for full on-device KV-cache

The engine does not yet implement an attention/KV-cache fast path because the op set it supports
(elementwise, activations, Transpose, Softmax, Reduce, MatMul, Conv) does not include the
transformer attention ops (no `Gather`/`Concat`/`Slice`/`Cast`/`LayerNormalization` on the GPU
engine — those run on the CPU engine only). A genuine on-device KV-cache needs, in order:

1. **GPU ops for the attention graph**: `Concat` (to append new K/V along the sequence axis),
   `Gather`/`Slice` (cache indexing), and ideally `LayerNormalization`, all as device→device
   kernels in `GpuKernels`. Until these exist, any attention graph falls back to the CPU engine
   and the question of on-device caching is moot.

2. **A persistent cross-`Run` device buffer pool.** Today every `Run` allocates fresh device
   buffers and disposes them in `finally`. An autoregressive KV-cache must instead hold the K/V
   buffers on the device **across decode steps** (across `Run` calls), appending one token's K/V
   per step rather than recomputing the whole prefix. That requires:
   - a cache object owning `MemoryBuffer1D` handles that outlive a single `Run`;
   - an "append" kernel/path that writes the new step's K/V at the current sequence offset of the
     persistent buffer (no realloc, no host copy);
   - lifetime/eviction management (max sequence length, reset between sequences).

3. **Engine API surface** to express "this is a decode step, reuse cache X" — the current
   `IExecutionEngine.Run(feeds) -> outputs` contract is stateless, so KV-cache needs either an
   overload that threads a cache handle through, or a stateful decode session wrapper.

None of (1)–(3) could be added safely within this change without touching the core engine seam
(owned by another agent) and adding several new GPU kernels — a much larger piece of work. The
safe, completed portion of B5 here is: **confirm intermediates already stay on-device, and remove
the one remaining mid-chain host round-trip (the Reduce identity copy).**
