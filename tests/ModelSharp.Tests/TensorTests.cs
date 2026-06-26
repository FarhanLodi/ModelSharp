using System;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class TensorTests
{
    [Fact]
    public void Shape_Length_Is_Product_Of_Dims()
    {
        var s = new TensorShape(2, 3, 4);
        Assert.Equal(24L, s.Length);
        Assert.Equal(3, s.Rank);
    }

    [Fact]
    public void Tensor_Rejects_Buffer_Of_Wrong_Length()
    {
        Assert.Throws<ArgumentException>(() => new Tensor<float>(new TensorShape(2, 2), new float[3]));
    }

    [Fact]
    public void Shapes_With_Same_Dims_Are_Equal()
    {
        Assert.True(new TensorShape(1, 3, 224, 224).Equals(new TensorShape(1, 3, 224, 224)));
        Assert.False(new TensorShape(1, 3, 224, 224).Equals(new TensorShape(1, 3, 256, 256)));
    }
}
