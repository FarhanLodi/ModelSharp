using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Generation;
using ModelSharp.Graph;
using ModelSharp.Manifest;
using ModelSharp.Onnx;

namespace ModelSharp.Pipeline;

/// <summary>
/// The headline entry point: load any model file and get a ready-to-run
/// <see cref="Pipeline"/>. Loads the ONNX graph, resolves a manifest
/// (sidecar JSON → embedded metadata → built-in heuristic), builds a
/// <see cref="ManagedCpuEngine"/>, and wires the pre/post processors selected for the
/// manifest's task — so <c>ModelSharpPipeline.Load("model.onnx").Run&lt;float[]&gt;("text")</c>
/// works out of the box for the built-in tasks.
/// </summary>
public static class ModelSharpPipeline
{
    /// <summary>Loads a model and resolves its manifest automatically.</summary>
    public static Pipeline Load(string modelPath)
    {
        if (modelPath is null) throw new ArgumentNullException(nameof(modelPath));
        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        ModelManifest manifest = ManifestResolver.Resolve(modelPath, graph);
        return Build(graph, manifest);
    }

    /// <summary>Loads a model using an explicit manifest, bypassing manifest resolution.</summary>
    public static Pipeline Load(string modelPath, ModelManifest manifest)
    {
        if (modelPath is null) throw new ArgumentNullException(nameof(modelPath));
        if (manifest is null) throw new ArgumentNullException(nameof(manifest));
        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        return Build(graph, manifest);
    }

    /// <summary>Builds a pipeline from an already-loaded graph and resolved manifest.</summary>
    public static Pipeline Build(ModelGraph graph, ModelManifest manifest)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (manifest is null) throw new ArgumentNullException(nameof(manifest));

        var engine = new ManagedCpuEngine(graph);
        List<string> inputNames = engine.Inputs.Select(i => i.Name).ToList();
        List<string> outputNames = engine.Outputs.Select(o => o.Name).ToList();
        var ctx = new ProcessorContext(manifest, inputNames, outputNames);

        // Text generation is autoregressive: it does not fit the single-shot pre/post path, so we
        // special-case it here (before ProcessorRegistry resolution) and wire a TextGenerator +
        // BPE tokenizer into the generation-flavoured Pipeline constructor. This keeps the registry
        // free of a TextGeneration entry — Generate / GenerateStream are the entry points, not Run<T>.
        if (manifest.Task == ModelTask.TextGeneration)
        {
            var generation = new TextGenerationProcessor(ctx);
            TextGenerator generator = generation.CreateGenerator(engine);
            return new Pipeline(engine, manifest, generator, generation);
        }

        IPreprocessor pre = ProcessorRegistry.CreatePreprocessor(ctx);
        IPostprocessor post = ProcessorRegistry.CreatePostprocessor(ctx);

        return new Pipeline(engine, manifest, pre, post);
    }
}
