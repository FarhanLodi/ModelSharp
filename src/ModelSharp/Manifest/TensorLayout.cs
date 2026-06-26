namespace ModelSharp.Manifest;

/// <summary>Memory layout of an image tensor.</summary>
public enum TensorLayout
{
    /// <summary>Batch, Channels, Height, Width (most ONNX vision models).</summary>
    Nchw,

    /// <summary>Batch, Height, Width, Channels (many TensorFlow-exported models).</summary>
    Nhwc,
}
