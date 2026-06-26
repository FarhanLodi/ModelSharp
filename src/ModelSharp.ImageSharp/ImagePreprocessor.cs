using System;
using System.Collections.Generic;
using System.IO;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ModelSharp.ImageSharp;

/// <summary>
/// Turns an image into a model feed tensor per the manifest: resize to W×H, scale to
/// [0,1], apply per-channel mean/std normalization, honor RGB/BGR order, and emit either
/// NCHW (shape [1, 3, H, W], the default) or NHWC (shape [1, H, W, 3]) according to
/// <see cref="ModelManifest.Layout"/>. Accepts an <see cref="Image{Rgb24}"/>, a file
/// path, raw bytes, or a stream.
/// </summary>
public sealed class ImagePreprocessor : IPreprocessor
{
    private readonly string _inputName;
    private readonly ModelManifest _manifest;

    public ImagePreprocessor(string inputName, ModelManifest manifest)
    {
        _inputName = inputName;
        _manifest = manifest;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, NamedTensor> ToFeeds(object input)
    {
        bool nhwc = _manifest.Layout == TensorLayout.Nhwc;
        if (!nhwc && _manifest.Layout != TensorLayout.Nchw)
            throw new NotSupportedException("ImagePreprocessor emits NCHW or NHWC only.");

        using Image<Rgb24> image = Load(input);
        int w = _manifest.Width > 0 ? _manifest.Width : image.Width;
        int h = _manifest.Height > 0 ? _manifest.Height : image.Height;
        if (image.Width != w || image.Height != h)
            image.Mutate(c => c.Resize(w, h));

        var data = new float[3 * h * w];
        IReadOnlyList<float> mean = _manifest.Mean, std = _manifest.Std;
        bool bgr = _manifest.Color == ColorOrder.Bgr;
        int hw = h * w;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    Rgb24 p = row[x];
                    float r = p.R / 255f, g = p.G / 255f, b = p.B / 255f;
                    float c0 = bgr ? b : r;   // channel 0
                    float c2 = bgr ? r : b;   // channel 2
                    float v0 = (c0 - mean[0]) / std[0];
                    float v1 = (g - mean[1]) / std[1];
                    float v2 = (c2 - mean[2]) / std[2];
                    if (nhwc)
                    {
                        // Channel-last: data[((y*w)+x)*3 + c].
                        int baseIdx = (y * w + x) * 3;
                        data[baseIdx + 0] = v0;
                        data[baseIdx + 1] = v1;
                        data[baseIdx + 2] = v2;
                    }
                    else
                    {
                        // Channel-first (NCHW): data[c*H*W + y*W + x].
                        data[0 * hw + y * w + x] = v0;
                        data[1 * hw + y * w + x] = v1;
                        data[2 * hw + y * w + x] = v2;
                    }
                }
            }
        });

        TensorShape shape = nhwc ? new TensorShape(1, h, w, 3) : new TensorShape(1, 3, h, w);
        var tensor = new Tensor<float>(shape, data);
        return new Dictionary<string, NamedTensor> { [_inputName] = new NamedTensor(_inputName, tensor) };
    }

    private static Image<Rgb24> Load(object input) => input switch
    {
        Image<Rgb24> img => img.Clone(),
        string path => Image.Load<Rgb24>(path),
        byte[] bytes => Image.Load<Rgb24>(bytes.AsSpan()),
        Stream stream => Image.Load<Rgb24>(stream),
        _ => throw new ArgumentException(
            $"Unsupported input type '{input?.GetType().Name ?? "null"}'. Expected Image<Rgb24>, string path, byte[], or Stream.",
            nameof(input)),
    };
}
