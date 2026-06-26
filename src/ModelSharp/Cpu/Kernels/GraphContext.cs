using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels;

/// <summary>The mutable name → tensor environment threaded through kernel execution.</summary>
public sealed class GraphContext
{
    private readonly Dictionary<string, Tensor> _values;

    public GraphContext(Dictionary<string, Tensor> values) => _values = values;

    /// <summary>
    /// Reads a tensor by name as float32, throwing if it hasn't been produced yet
    /// or if its dtype is not Float32. Existing float kernels use this overload.
    /// </summary>
    public Tensor<float> Get(string name) => GetTensor(name).AsFloat();

    /// <summary>
    /// Reads a tensor by name preserving its dtype, throwing if it hasn't been
    /// produced yet. Dtype-aware kernels use this and then call
    /// <see cref="Tensor.AsInt64"/>/<see cref="Tensor.AsBool"/>/etc.
    /// </summary>
    public Tensor GetTensor(string name) =>
        _values.TryGetValue(name, out Tensor? t)
            ? t
            : throw new KeyNotFoundException($"Tensor '{name}' is not available in the execution context.");

    /// <summary>Writes a tensor by name (any dtype; <c>Tensor&lt;float&gt;</c> upcasts implicitly).</summary>
    public void Set(string name, Tensor value) => _values[name] = value;
}
