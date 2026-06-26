namespace ModelSharp;

/// <summary>
/// Thrown when a model uses an operator the managed engine doesn't implement yet.
/// The message names the exact op (and node) so coverage gaps are obvious and the
/// next kernel to write is unambiguous.
/// </summary>
public sealed class UnsupportedOperatorException : ModelSharpException
{
    /// <summary>The ONNX op type that is unsupported.</summary>
    public string Operator { get; }

    public UnsupportedOperatorException(string op, string? detail = null)
        : base($"Operator '{op}' is not supported yet by the managed engine"
               + (detail is null ? "." : $" ({detail})."))
    {
        Operator = op;
    }
}
