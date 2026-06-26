using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Pipeline;

/// <summary>Turns a typed input (image, text, audio) into model feed tensors.</summary>
public interface IPreprocessor
{
    /// <summary>Converts the input into named feed tensors for the engine.</summary>
    IReadOnlyDictionary<string, NamedTensor> ToFeeds(object input);
}
