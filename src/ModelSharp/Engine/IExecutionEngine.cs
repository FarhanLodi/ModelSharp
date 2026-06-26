using System;
using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Engine;

/// <summary>
/// A backend that executes a model graph. Implementations: managed CPU (Phase 1),
/// ILGPU GPU (later). The pipeline talks only to this interface, so engines are
/// swappable and the public API never changes as the engine matures.
/// </summary>
public interface IExecutionEngine : IDisposable
{
    /// <summary>The model's input bindings.</summary>
    IReadOnlyList<TensorInfo> Inputs { get; }

    /// <summary>The model's output bindings.</summary>
    IReadOnlyList<TensorInfo> Outputs { get; }

    /// <summary>Runs the graph for the supplied named feeds and returns the named outputs.</summary>
    IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds);
}
