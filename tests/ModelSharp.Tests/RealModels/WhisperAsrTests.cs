using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Audio;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Generation;
using ModelSharp.Graph;
using ModelSharp.Manifest;
using ModelSharp.Onnx;
using ModelSharp.Pipeline;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Opt-in integration test against a real Whisper speech-to-text export (e.g. whisper-tiny / whisper-base,
/// the Optimum two-file form: an audio encoder + a decoder-with-past/merged decoder). No-ops (green) unless
/// the model + tokenizer + audio files are present in the resolved models dir; becomes live when a user
/// drops them in.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>whisper-encoder.onnx</c>, <c>whisper-decoder.onnx</c>, <c>whisper-vocab.json</c>,
/// <c>whisper-merges.txt</c>, and a mono 16 kHz <c>whisper-audio.wav</c>. The special-token ids and
/// KV head dims below match the standard multilingual whisper-tiny/base checkpoints.</para>
/// </summary>
public class WhisperAsrTests
{
    private readonly ITestOutputHelper _out;
    public WhisperAsrTests(ITestOutputHelper output) => _out = output;

    private const string EncoderFile = "whisper-encoder.onnx";
    private const string DecoderFile = "whisper-decoder.onnx";
    private const string VocabFile = "whisper-vocab.json";
    private const string MergesFile = "whisper-merges.txt";
    private const string AudioFile = "whisper-audio.wav";

    // whisper-tiny: 6 decoder heads, head_dim 64. Multilingual special-token block (vocab 51865):
    // <|endoftext|>=50257, <|startoftranscript|>=50258, <|en|>=50259, <|transcribe|>=50359, <|notimestamps|>=50363.
    private const int NumHeads = 6;
    private const int HeadDim = 64;

    [Fact]
    public void Whisper_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath))
        {
            _out.WriteLine("whisper model not present; skipping.");
            return;
        }

        KernelRegistry registry = KernelRegistry.CreateDefault();
        foreach (string path in new[] { encPath, decPath })
        {
            ModelGraph g = OnnxModelLoader.LoadModel(path);
            var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
            var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();
            _out.WriteLine($"{path}: nodes={g.Nodes.Count} distinctOps={distinct.Count}");
            _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
            Assert.True(missing.Count == 0, $"Unsupported ops in {path}: " + string.Join(", ", missing));
        }
    }

    [Fact]
    public void Whisper_Transcribes_Recognizably_AndIsDeterministic()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath)
            || !RealModelAssets.TryPath(MergesFile, out string mergesPath)
            || !RealModelAssets.TryPath(AudioFile, out string audioPath))
        {
            _out.WriteLine("whisper assets not present; skipping.");
            return;
        }

        var manifest = new ModelManifest
        {
            Task = ModelTask.SpeechToTextSeq2Seq,
            Extra = new Dictionary<string, string>
            {
                ["vocab"] = vocabPath,
                ["merges"] = mergesPath,
                ["n_mels"] = "80",
                ["language"] = "50259",   // <|en|>
                ["task"] = "transcribe",
                ["kv_num_heads"] = NumHeads.ToString(),
                ["kv_head_dim"] = HeadDim.ToString(),
                ["max_new_tokens"] = "64",
            },
        };

        float[] waveform = ReadWavMono16(audioPath);

        using var pipeline = ModelSharpPipeline.LoadWhisper(encPath, decPath, manifest);
        string first = pipeline.Transcribe(waveform);
        _out.WriteLine($"transcript: '{first}'");

        Assert.False(string.IsNullOrWhiteSpace(first), "Whisper produced no transcription.");
        Assert.Contains(first, c => char.IsLetter(c));

        // Greedy decoding is deterministic: the same audio reproduces exactly.
        using var pipeline2 = ModelSharpPipeline.LoadWhisper(encPath, decPath, manifest);
        string second = pipeline2.Transcribe(waveform);
        Assert.Equal(first, second);
    }

    /// <summary>Minimal mono 16-bit PCM WAV reader → float samples in [-1, 1] (mirrors the wav2vec2 test).</summary>
    private static float[] ReadWavMono16(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        // Locate the "data" chunk (skip the RIFF/fmt headers without assuming a fixed offset).
        int p = 12;
        int dataOffset = -1, dataLen = 0, channels = 1, bits = 16;
        while (p + 8 <= bytes.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(bytes, p, 4);
            int size = BitConverter.ToInt32(bytes, p + 4);
            if (id == "fmt ")
            {
                channels = BitConverter.ToInt16(bytes, p + 10);
                bits = BitConverter.ToInt16(bytes, p + 22);
            }
            else if (id == "data")
            {
                dataOffset = p + 8;
                dataLen = size;
                break;
            }
            p += 8 + size + (size & 1);
        }
        if (dataOffset < 0) throw new InvalidDataException("No data chunk in WAV.");
        if (bits != 16) throw new NotSupportedException($"Only 16-bit PCM WAV supported; got {bits}-bit.");

        int bytesPerSample = 2 * channels;
        int frames = dataLen / bytesPerSample;
        var samples = new float[frames];
        for (int i = 0; i < frames; i++)
        {
            int off = dataOffset + i * bytesPerSample; // take channel 0 only
            short s = BitConverter.ToInt16(bytes, off);
            samples[i] = s / 32768f;
        }
        return samples;
    }
}
