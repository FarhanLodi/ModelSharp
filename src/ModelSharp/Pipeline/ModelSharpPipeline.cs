using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Engine;
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

    /// <summary>
    /// Loads a two-file encoder-decoder (seq2seq) model — the standard HF/Optimum export shape
    /// (<c>encoder_model.onnx</c> + <c>decoder_model.onnx</c> / <c>decoder_with_past_model.onnx</c> or the
    /// merged decoder) — and wires a <see cref="Generation.Seq2SeqGenerator"/> over both graphs. The
    /// manifest's task must be <see cref="ModelTask.Seq2SeqGeneration"/>.
    /// </summary>
    /// <param name="encoderPath">Path to the encoder ONNX file.</param>
    /// <param name="decoderPath">Path to the decoder ONNX file (with-past or merged form is preferred for speed).</param>
    /// <param name="manifest">The resolved manifest (task = Seq2SeqGeneration, tokenizer + start-token hints under Extra).</param>
    public static Pipeline LoadSeq2Seq(string encoderPath, string decoderPath, ModelManifest manifest)
    {
        if (encoderPath is null) throw new ArgumentNullException(nameof(encoderPath));
        if (decoderPath is null) throw new ArgumentNullException(nameof(decoderPath));
        if (manifest is null) throw new ArgumentNullException(nameof(manifest));
        if (manifest.Task != ModelTask.Seq2SeqGeneration)
            throw new ModelSharpException(
                $"LoadSeq2Seq requires a manifest with Task = Seq2SeqGeneration; got '{manifest.Task}'.");

        ModelGraph encoderGraph = OnnxModelLoader.LoadModel(encoderPath);
        ModelGraph decoderGraph = OnnxModelLoader.LoadModel(decoderPath);
        var encoderEngine = new ManagedCpuEngine(encoderGraph);
        var decoderEngine = new ManagedCpuEngine(decoderGraph);
        return BuildSeq2Seq(encoderEngine, decoderEngine, manifest);
    }

    /// <summary>Builds a seq2seq pipeline from already-constructed encoder and decoder engines.</summary>
    public static Pipeline BuildSeq2Seq(IExecutionEngine encoderEngine, IExecutionEngine decoderEngine, ModelManifest manifest)
    {
        if (encoderEngine is null) throw new ArgumentNullException(nameof(encoderEngine));
        if (decoderEngine is null) throw new ArgumentNullException(nameof(decoderEngine));
        if (manifest is null) throw new ArgumentNullException(nameof(manifest));

        List<string> inputNames = decoderEngine.Inputs.Select(i => i.Name).ToList();
        List<string> outputNames = decoderEngine.Outputs.Select(o => o.Name).ToList();
        var ctx = new ProcessorContext(manifest, inputNames, outputNames);

        var generation = new Seq2SeqGenerationProcessor(ctx);
        Seq2SeqGenerator generator = generation.CreateGenerator(encoderEngine, decoderEngine);
        return new Pipeline(decoderEngine, encoderEngine, manifest, generator, generation);
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

        // Seq2seq (encoder-decoder) from a single combined graph: the same engine serves both the encoder
        // and decoder roles. The standard two-file export uses ModelSharpPipeline.LoadSeq2Seq instead.
        if (manifest.Task == ModelTask.Seq2SeqGeneration)
            return BuildSeq2Seq(engine, engine, manifest);

        IPreprocessor pre = ProcessorRegistry.CreatePreprocessor(ctx);
        IPostprocessor post = ProcessorRegistry.CreatePostprocessor(ctx);

        return new Pipeline(engine, manifest, pre, post);
    }
}
