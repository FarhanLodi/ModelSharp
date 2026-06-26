using System.Collections.Generic;
using System.Linq;
using ModelSharp.ImageSharp;
using ModelSharp.Manifest;
using ModelSharp.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace ModelSharp.Tests;

public class ImagePreprocessorTests
{
    [Fact]
    public void Red_Image_Produces_Expected_Nchw_Tensor()
    {
        var manifest = new ModelManifest { Width = 2, Height = 2, Layout = TensorLayout.Nchw };
        var pre = new ImagePreprocessor("input", manifest);

        using var img = new Image<Rgb24>(2, 2, new Rgb24(255, 0, 0));
        IReadOnlyDictionary<string, NamedTensor> feeds = pre.ToFeeds(img);

        Tensor<float> t = feeds["input"].Data;
        Assert.Equal(new[] { 1, 3, 2, 2 }, t.Shape.Dimensions.ToArray());

        float[] v = t.Span.ToArray();
        Assert.All(v.Take(4), c => Assert.Equal(1f, c, 5));   // R channel
        Assert.All(v.Skip(4), c => Assert.Equal(0f, c, 5));   // G + B channels
    }

    [Fact]
    public void Red_Image_Produces_Expected_Nhwc_Tensor()
    {
        var manifest = new ModelManifest { Width = 2, Height = 2, Layout = TensorLayout.Nhwc };
        var pre = new ImagePreprocessor("input", manifest);

        using var img = new Image<Rgb24>(2, 2, new Rgb24(255, 0, 0));
        IReadOnlyDictionary<string, NamedTensor> feeds = pre.ToFeeds(img);

        Tensor<float> t = feeds["input"].Data;
        Assert.Equal(new[] { 1, 2, 2, 3 }, t.Shape.Dimensions.ToArray());

        // Channel-last: data[((y*w)+x)*3 + c] => [r, g, b] repeated for each of the 4 pixels.
        var expected = new[]
        {
            1f, 0f, 0f,   // pixel (y0, x0)
            1f, 0f, 0f,   // pixel (y0, x1)
            1f, 0f, 0f,   // pixel (y1, x0)
            1f, 0f, 0f,   // pixel (y1, x1)
        };

        float[] v = t.Span.ToArray();
        Assert.Equal(expected.Length, v.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], v[i], 5);
    }

    [Fact]
    public void Nhwc_Honors_Bgr_Order()
    {
        var manifest = new ModelManifest
        {
            Width = 1,
            Height = 1,
            Layout = TensorLayout.Nhwc,
            Color = ColorOrder.Bgr,
        };
        var pre = new ImagePreprocessor("input", manifest);

        using var img = new Image<Rgb24>(1, 1, new Rgb24(255, 0, 0));   // pure red
        IReadOnlyDictionary<string, NamedTensor> feeds = pre.ToFeeds(img);

        Tensor<float> t = feeds["input"].Data;
        Assert.Equal(new[] { 1, 1, 1, 3 }, t.Shape.Dimensions.ToArray());

        // BGR channel-last for a pure-red pixel: [b, g, r] = [0, 0, 1].
        float[] v = t.Span.ToArray();
        Assert.Equal(0f, v[0], 5);
        Assert.Equal(0f, v[1], 5);
        Assert.Equal(1f, v[2], 5);
    }
}
