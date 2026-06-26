namespace ModelSharp.Tensors;

/// <summary>A tensor paired with the graph binding name it feeds or produces.</summary>
public sealed class NamedTensor
{
    /// <summary>The graph input/output name.</summary>
    public string Name { get; }

    /// <summary>The dtype-carrying tensor payload.</summary>
    public Tensor Tensor { get; }

    /// <summary>The payload as a float32 tensor (convenience; throws if the dtype is not Float32).</summary>
    public Tensor<float> Data => Tensor.AsFloat();

    /// <summary>Binds a dtype-carrying tensor to a name.</summary>
    public NamedTensor(string name, Tensor data)
    {
        Name = name;
        Tensor = data;
    }
}
