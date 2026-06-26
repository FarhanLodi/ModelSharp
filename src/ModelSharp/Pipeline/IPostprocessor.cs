using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Pipeline;

/// <summary>Turns raw model outputs into a typed, user-facing result.</summary>
public interface IPostprocessor
{
    /// <summary>Decodes engine outputs into a typed result object.</summary>
    object Decode(IReadOnlyDictionary<string, NamedTensor> outputs);
}
