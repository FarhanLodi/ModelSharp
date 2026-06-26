using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using ModelSharp.Audio;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// A4 — Opt-in integration test against a real wav2vec2 CTC ONNX model.
/// No-ops (green) unless the model + audio + vocab are present in the resolved models dir.
/// See <c>docs/REAL_MODELS.md</c> for the export recipe.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>wav2vec2.onnx</c> (CTC head; input <c>input_values</c> of raw 16&#160;kHz mono waveform,
/// output <c>logits</c> of shape <c>[1, T, V]</c>), <c>speech.wav</c> (a short mono 16&#160;kHz PCM-16
/// clip), and <c>wav2vec2-vocab.json</c> (the HF <c>{token: index}</c> map).</para>
///
/// <para>wav2vec2's own convolutional feature extractor consumes the raw normalized waveform, so no
/// mel front end is applied here; the blank symbol is index 0 (the standard wav2vec2 convention,
/// shared with <c>&lt;pad&gt;</c>).</para>
/// </summary>
public class Wav2Vec2CtcTests
{
    private readonly ITestOutputHelper _out;
    public Wav2Vec2CtcTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "wav2vec2.onnx";
    private const string AudioFile = "speech.wav";
    private const string VocabFile = "wav2vec2-vocab.json";
    private const string InputName = "input_values";
    private const int Blank = 0;

    [Fact]
    public void Wav2Vec2_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"wav2vec2 model not present ({modelPath}); skipping.");
            return;
        }

        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        _out.WriteLine($"nodes={g.Nodes.Count}  distinctOps={distinct.Count}  initializers={g.Initializers.Count}");
        _out.WriteLine("ALL OPS: " + string.Join(", ", distinct));
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
    }

    [Fact]
    public void Wav2Vec2_Transcribes_Recognizably()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(AudioFile, out string audioPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath))
        {
            _out.WriteLine("wav2vec2 assets not present; skipping.");
            return;
        }

        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        CtcVocabulary vocabulary = LoadVocabulary(vocabPath);

        // Raw mono waveform, zero-mean / unit-variance normalized (the wav2vec2 feature extractor
        // convention), fed as input_values [1, samples].
        float[] samples = ReadWavMono16(audioPath);
        Normalize(samples);

        using var engine = new ManagedCpuEngine(graph);
        string inName = engine.Inputs.Count == 1 ? engine.Inputs[0].Name : InputName;
        var feeds = new Dictionary<string, NamedTensor>
        {
            [inName] = new NamedTensor(inName, new Tensor<float>(new TensorShape(1, samples.Length), samples)),
        };

        Tensor<float> logits = engine.Run(feeds).Values.Single().Data; // [1, T, V] (or [T, V])
        Tensor<float> emissions = ToTimeVocab(logits);

        string transcript = CtcDecoder.GreedyDecode(emissions, vocabulary, Blank).Trim();
        _out.WriteLine($"frames={emissions.Shape[0]}  vocab={emissions.Shape[1]}");
        _out.WriteLine($"transcript: '{transcript}'");

        Assert.False(string.IsNullOrWhiteSpace(transcript), "CTC decode produced an empty transcript.");
        // A recognizable transcription contains at least one real word (letters, not just punctuation).
        Assert.Contains(transcript, c => char.IsLetter(c));
    }

    /// <summary>Builds a <see cref="CtcVocabulary"/> from a HF <c>{token: index}</c> vocab.json.</summary>
    private static CtcVocabulary LoadVocabulary(string vocabJsonPath)
    {
        Dictionary<string, int> map = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabJsonPath))
            ?? throw new InvalidDataException($"Failed to parse wav2vec2 vocab JSON: {vocabJsonPath}");
        int size = map.Values.Max() + 1;
        var ordered = new string[size];
        foreach (KeyValuePair<string, int> kv in map) ordered[kv.Value] = kv.Key;
        for (int i = 0; i < size; i++) ordered[i] ??= string.Empty;
        // wav2vec2 uses "|" as the word delimiter and <pad>/<s>/</s>/<unk> as specials (defaults).
        return new CtcVocabulary(ordered, wordDelimiter: "|");
    }

    /// <summary>Collapses a <c>[1, T, V]</c> logits tensor to the rank-2 <c>[T, V]</c> CtcDecoder expects.</summary>
    private static Tensor<float> ToTimeVocab(Tensor<float> logits)
    {
        ReadOnlySpan<int> dims = logits.Shape.Dimensions;
        if (dims.Length == 2) return logits;
        if (dims.Length == 3 && dims[0] == 1)
            return (Tensor<float>)logits.WithShape(new TensorShape(dims[1], dims[2]));
        throw new InvalidDataException($"Unexpected wav2vec2 logits shape {logits.Shape}; expected [1, T, V] or [T, V].");
    }

    /// <summary>Zero-mean, unit-variance normalization in place (matches the wav2vec2 feature extractor).</summary>
    private static void Normalize(float[] x)
    {
        if (x.Length == 0) return;
        double mean = 0;
        for (int i = 0; i < x.Length; i++) mean += x[i];
        mean /= x.Length;
        double var = 0;
        for (int i = 0; i < x.Length; i++) { double d = x[i] - mean; var += d * d; }
        var /= x.Length;
        float std = (float)Math.Sqrt(var) + 1e-7f;
        for (int i = 0; i < x.Length; i++) x[i] = (float)((x[i] - mean) / std);
    }

    /// <summary>
    /// Minimal RIFF/WAVE reader for mono 16-bit PCM, returning samples in [-1, 1].
    /// Stereo input is averaged to mono; only the canonical PCM-16 layout is supported.
    /// </summary>
    private static float[] ReadWavMono16(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 44 || bytes[0] != 'R' || bytes[1] != 'I' || bytes[2] != 'F' || bytes[3] != 'F'
            || bytes[8] != 'W' || bytes[9] != 'A' || bytes[10] != 'V' || bytes[11] != 'E')
            throw new InvalidDataException($"Not a RIFF/WAVE file: {path}");

        int channels = 1;
        int bitsPerSample = 16;
        int pos = 12;
        int dataOffset = -1, dataLength = 0;
        while (pos + 8 <= bytes.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(bytes, pos, 4);
            int size = BitConverter.ToInt32(bytes, pos + 4);
            int body = pos + 8;
            if (id == "fmt ")
            {
                channels = BitConverter.ToInt16(bytes, body + 2);
                bitsPerSample = BitConverter.ToInt16(bytes, body + 14);
            }
            else if (id == "data")
            {
                dataOffset = body;
                dataLength = size;
                break;
            }
            pos = body + size + (size & 1); // chunks are word-aligned
        }
        if (dataOffset < 0) throw new InvalidDataException($"No data chunk in WAV: {path}");
        if (bitsPerSample != 16) throw new NotSupportedException($"Only 16-bit PCM WAV is supported; got {bitsPerSample}-bit.");
        dataLength = Math.Min(dataLength, bytes.Length - dataOffset);

        int totalSamples = dataLength / 2;
        int frames = channels > 0 ? totalSamples / channels : totalSamples;
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            int acc = 0;
            for (int c = 0; c < channels; c++)
            {
                short s = BitConverter.ToInt16(bytes, dataOffset + (f * channels + c) * 2);
                acc += s;
            }
            mono[f] = acc / (channels * 32768f);
        }
        return mono;
    }
}
