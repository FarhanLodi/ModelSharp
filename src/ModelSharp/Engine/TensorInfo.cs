using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Engine;

/// <summary>Static description of a model input or output binding.</summary>
public sealed class TensorInfo
{
    /// <summary>The binding name.</summary>
    public string Name { get; }

    /// <summary>The element type of the binding.</summary>
    public ElementType ElementType { get; }

    /// <summary>Declared dimensions; <c>-1</c> marks a dynamic axis (e.g. batch).</summary>
    public IReadOnlyList<int> Dimensions { get; }

    public TensorInfo(string name, ElementType elementType, IReadOnlyList<int> dimensions)
    {
        Name = name;
        ElementType = elementType;
        Dimensions = dimensions;
    }
}
