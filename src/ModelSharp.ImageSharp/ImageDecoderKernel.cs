using System;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ModelSharp.ImageSharp;

/// <summary>
/// ONNX <c>ImageDecoder</c> (opset 20): decodes a 1-D <c>uint8</c> tensor of encoded image bytes
/// (JPEG / PNG / BMP / …) into a <c>uint8</c> pixel tensor of shape <c>[H, W, C]</c> using
/// SixLabors.ImageSharp's format-sniffing <c>Image.Load</c>.
///
/// <para>
/// The <c>pixel_format</c> attribute selects the channel layout (ONNX default is <c>"RGB"</c>):
/// <list type="bullet">
/// <item><c>"RGB"</c> — 3-channel, output[...,0..2] = R, G, B.</item>
/// <item><c>"BGR"</c> — 3-channel, output[...,0..2] = B, G, R.</item>
/// <item><c>"Grayscale"</c> — 1-channel luminance (output shape <c>[H, W, 1]</c>).</item>
/// </list>
/// </para>
///
/// <para>
/// This kernel lives in the ImageSharp adapter (which already depends on SixLabors.ImageSharp);
/// register it on a <see cref="KernelRegistry"/> via
/// <see cref="ImageSharpRegistration.RegisterKernels(KernelRegistry)"/> so the core stays free of
/// an image-codec dependency.
/// </para>
/// </summary>
public sealed class ImageDecoderKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "ImageDecoder";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor encoded = ctx.GetTensor(node.Inputs[0]);
        if (encoded.Dtype != ElementType.UInt8)
            throw new ModelSharpException(
                $"ImageDecoder node '{node.Name}': encoded-bytes input must be uint8, got {encoded.Dtype}.");

        // The encoded bytes are a flat uint8 tensor (any rank; row-major buffer is the byte stream).
        ReadOnlySpan<byte> bytes = ((Tensor<byte>)encoded).Span;

        string format = ReadStringAttr(node, "pixel_format", "RGB");
        bool grayscale = string.Equals(format, "Grayscale", StringComparison.OrdinalIgnoreCase);
        bool bgr = string.Equals(format, "BGR", StringComparison.OrdinalIgnoreCase);
        if (!grayscale && !bgr && !string.Equals(format, "RGB", StringComparison.OrdinalIgnoreCase))
            throw new ModelSharpException(
                $"ImageDecoder node '{node.Name}': unsupported pixel_format '{format}' "
                + "(expected 'RGB', 'BGR', or 'Grayscale').");

        using Image<Rgb24> image = Image.Load<Rgb24>(bytes);
        int h = image.Height, w = image.Width;
        int channels = grayscale ? 1 : 3;

        var data = new byte[h * w * channels];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                int rowBase = y * w * channels;
                for (int x = 0; x < w; x++)
                {
                    Rgb24 p = row[x];
                    int o = rowBase + x * channels;
                    if (grayscale)
                    {
                        // ITU-R BT.601 luma, matching common ONNX runtimes' grayscale conversion.
                        data[o] = (byte)((p.R * 299 + p.G * 587 + p.B * 114 + 500) / 1000);
                    }
                    else if (bgr)
                    {
                        data[o + 0] = p.B;
                        data[o + 1] = p.G;
                        data[o + 2] = p.R;
                    }
                    else
                    {
                        data[o + 0] = p.R;
                        data[o + 1] = p.G;
                        data[o + 2] = p.B;
                    }
                }
            }
        });

        var output = new Tensor<byte>(new TensorShape(h, w, channels), data);
        ctx.Set(node.Outputs[0], output);
    }

    /// <summary>Reads a STRING attribute (boxed as <see cref="string"/>) with a default.</summary>
    private static string ReadStringAttr(GraphNode node, string name, string dflt)
        => node.Attributes.TryGetValue(name, out object? v) && v is string s ? s : dflt;
}
