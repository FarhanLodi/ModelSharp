using System.Runtime.CompilerServices;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;

namespace ModelSharp.ImageSharp;

/// <summary>
/// Wires the ImageSharp processors into the core <see cref="ProcessorRegistry"/> so a
/// manifest-driven pipeline can auto-build them for image tasks. Registration happens
/// in a <see cref="ModuleInitializerAttribute"/> that fires once this assembly is
/// touched; call <see cref="Ensure"/> to guarantee it has run.
/// </summary>
public static class ImageSharpRegistration
{
    /// <summary>
    /// Registers the image-classification pre/post-processor factories with the core
    /// <see cref="ProcessorRegistry"/>. Invoked automatically when the assembly loads.
    /// </summary>
#pragma warning disable CA2255 // ModuleInitializer is deliberate here: auto-register the image processors on assembly load.
    [ModuleInitializer]
    internal static void Initialize()
    {
        ProcessorRegistry.RegisterPreprocessor(
            ModelTask.ImageClassification,
            ctx => new ImagePreprocessor(ctx.InputNames[0], ctx.Manifest));

        ProcessorRegistry.RegisterPostprocessor(
            ModelTask.ImageClassification,
            ctx => new ClassificationPostprocessor(ctx.Manifest));
    }
#pragma warning restore CA2255

    /// <summary>
    /// No-op that guarantees the module initializer has executed. Because module
    /// initializers only fire once a member of the assembly is referenced, callers can
    /// invoke this before building a pipeline to force the registrations above to run.
    /// </summary>
    public static void Ensure() { }
}
