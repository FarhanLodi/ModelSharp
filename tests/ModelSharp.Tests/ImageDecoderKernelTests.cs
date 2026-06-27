using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Graph;
using ModelSharp.ImageSharp;
using ModelSharp.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the ImageSharp <see cref="ImageDecoderKernel"/> (ONNX <c>ImageDecoder</c>, opset 20):
/// a uint8 1-D tensor of encoded image bytes decodes to a uint8 <c>[H, W, C]</c> pixel tensor,
/// honoring the <c>pixel_format</c> attribute (RGB / BGR / Grayscale).
/// </summary>
public class ImageDecoderKernelTests
{
    /// <summary>Builds a known 2x2 RGB image with four distinct pixels and encodes it to PNG bytes.</summary>
    private static byte[] EncodeKnownPng()
    {
        using var img = new Image<Rgb24>(2, 2);
        img[0, 0] = new Rgb24(10, 20, 30);     // (x0, y0)
        img[1, 0] = new Rgb24(40, 50, 60);     // (x1, y0)
        img[0, 1] = new Rgb24(70, 80, 90);     // (x0, y1)
        img[1, 1] = new Rgb24(100, 110, 120);  // (x1, y1)

        using var ms = new MemoryStream();
        img.Save(ms, new PngEncoder());
        return ms.ToArray();
    }

    private static Tensor<byte> Decode(byte[] png, string? pixelFormat)
    {
        var attrs = new Dictionary<string, object>();
        if (pixelFormat is not null) attrs["pixel_format"] = pixelFormat;

        var node = new GraphNode("ImageDecoder", "dec", new[] { "encoded" }, new[] { "pixels" }, attrs);
        var env = new Dictionary<string, Tensor>
        {
            ["encoded"] = new Tensor<byte>(new TensorShape(png.Length), png),
        };
        var ctx = new GraphContext(env);

        new ImageDecoderKernel().Execute(node, ctx);
        return (Tensor<byte>)ctx.GetTensor("pixels");
    }

    [Fact]
    public void Decodes_Png_To_Rgb_Hwc_With_Known_Pixels()
    {
        Tensor<byte> px = Decode(EncodeKnownPng(), pixelFormat: null);  // default RGB

        Assert.Equal(new[] { 2, 2, 3 }, px.Shape.Dimensions.ToArray());
        // Row-major [H, W, C]: pixel (y, x) channels at ((y*W + x)*3 + c).
        Assert.Equal(
            new byte[]
            {
                10, 20, 30,     40, 50, 60,      // y0: (x0), (x1)
                70, 80, 90,     100, 110, 120,   // y1: (x0), (x1)
            },
            px.Span.ToArray());
    }

    [Fact]
    public void Honors_Bgr_Pixel_Format()
    {
        Tensor<byte> px = Decode(EncodeKnownPng(), pixelFormat: "BGR");

        Assert.Equal(new[] { 2, 2, 3 }, px.Shape.Dimensions.ToArray());
        // Same pixels, channels reversed to B, G, R.
        Assert.Equal(
            new byte[]
            {
                30, 20, 10,     60, 50, 40,
                90, 80, 70,     120, 110, 100,
            },
            px.Span.ToArray());
    }

    [Fact]
    public void Honors_Grayscale_Pixel_Format()
    {
        Tensor<byte> px = Decode(EncodeKnownPng(), pixelFormat: "Grayscale");

        Assert.Equal(new[] { 2, 2, 1 }, px.Shape.Dimensions.ToArray());

        // BT.601 luma: round((R*299 + G*587 + B*114) / 1000).
        static byte Luma(int r, int g, int b) => (byte)((r * 299 + g * 587 + b * 114 + 500) / 1000);
        Assert.Equal(
            new byte[]
            {
                Luma(10, 20, 30),  Luma(40, 50, 60),
                Luma(70, 80, 90),  Luma(100, 110, 120),
            },
            px.Span.ToArray());
    }

    [Fact]
    public void Runs_Through_Engine_With_Registered_Kernel()
    {
        // End-to-end: register the ImageSharp kernel on a default registry, then run a one-node
        // graph through ManagedCpuEngine. Proves the registration entry-point wires it up.
        KernelRegistry registry = ImageSharpRegistration.RegisterKernels(KernelRegistry.CreateDefault());

        var graph = new ModelGraph
        {
            Inputs = new[] { "encoded" },
            Outputs = new[] { "pixels" },
            Nodes = new[]
            {
                new GraphNode("ImageDecoder", "dec", new[] { "encoded" }, new[] { "pixels" }),
            },
        };

        byte[] png = EncodeKnownPng();
        using var engine = new ManagedCpuEngine(graph, registry);
        IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["encoded"] = new NamedTensor("encoded", new Tensor<byte>(new TensorShape(png.Length), png)),
        });

        var px = (Tensor<byte>)outputs["pixels"].Tensor;
        Assert.Equal(new[] { 2, 2, 3 }, px.Shape.Dimensions.ToArray());
        Assert.Equal((byte)10, px.Span[0]);
        Assert.Equal((byte)120, px.Span[^1]);
    }
}
