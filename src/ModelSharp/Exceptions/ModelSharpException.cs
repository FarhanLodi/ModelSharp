using System;

namespace ModelSharp;

/// <summary>Base type for all ModelSharp errors.</summary>
public class ModelSharpException : Exception
{
    public ModelSharpException(string message) : base(message) { }

    public ModelSharpException(string message, Exception innerException) : base(message, innerException) { }
}
