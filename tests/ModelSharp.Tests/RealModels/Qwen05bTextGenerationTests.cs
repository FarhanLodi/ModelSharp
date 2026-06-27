using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Engine;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using ModelSharp.Text;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// End-to-end "real decoded text" proof for Qwen2.5-0.5B-Instruct (INT4 q4 export): load the Hugging
/// Face <c>tokenizer.json</c> via <see cref="HfTokenizerJson"/>, encode a prompt to ids, run a short
/// greedy generation loop through the managed CPU engine (feeding the growing KV cache), and decode the
/// generated ids back to actual text. Asset-gated: skips cleanly when the model / tokenizer aren't present.
///
/// <para>Model shape: 24 layers, 2 KV heads, head_dim 64 (the same dims <see cref="Qwen05bInt4Tests"/>
/// documents). Inputs: <c>input_ids</c>, <c>attention_mask</c>, <c>position_ids</c> and
/// <c>past_key_values.N.{key,value}</c>; outputs: <c>logits</c> and <c>present.N.{key,value}</c>.</para>
/// </summary>
public class Qwen05bTextGenerationTests
{
    private readonly ITestOutputHelper _out;
    public Qwen05bTextGenerationTests(ITestOutputHelper output) => _out = output;

    private const string ModelRel = "qwen05b-q4/model_q4.onnx";
    private const string TokenizerRel = "qwen05b-q4/tokenizer.json";
    private const string HubSpec = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx";
    private const int KvHeads = 2, HeadDim = 64;

    /// <summary>
    /// Resolves the Qwen <c>tokenizer.json</c>: local first, otherwise (opt-in) via the Hub bundle that
    /// <see cref="HubSpec"/> downloads — the <c>tokenizer.json</c> companion lands next to the model in the
    /// Hub cache dir, so we look there. Skips cleanly (returns false) when neither is available.
    /// </summary>
    private bool TryResolveTokenizer(out string tokPath)
    {
        if (RealModelAssets.TryPath(TokenizerRel, out tokPath))
            return true;
        // The model bundle download also fetches tokenizer.json into the same directory.
        if (RealModelAssets.TryResolveOrDownload(ModelRel, HubSpec, out string modelPath, log: _out.WriteLine))
        {
            string? dir = System.IO.Path.GetDirectoryName(modelPath);
            if (dir is not null)
            {
                string candidate = System.IO.Path.Combine(dir, "tokenizer.json");
                if (System.IO.File.Exists(candidate)) { tokPath = candidate; return true; }
            }
        }
        return false;
    }
    // Greedy stop tokens (generation_config: <|im_end|> and <|endoftext|>).
    private static readonly HashSet<int> EosIds = new() { 151643, 151645 };

    // ---- pure tokenizer round-trip (no model needed) ----

    [Fact]
    public void HfTokenizer_RoundTrips_Real_Qwen_Text()
    {
        if (!TryResolveTokenizer(out string tokPath))
        {
            _out.WriteLine("Qwen tokenizer.json not found; skipping.");
            return;
        }

        BpeTokenizer tok = HfTokenizerJson.FromFile(tokPath);
        _out.WriteLine($"Loaded HF tokenizer: VocabSize={tok.VocabSize}");

        foreach (string text in new[]
        {
            "The capital of France is",
            " Hello, world! 12345",
            "don't worry it's fine",
            "héllo \U0001F600 世界 café", // accents, emoji, CJK
        })
        {
            int[] ids = tok.Encode(text);
            string back = tok.Decode(ids);
            _out.WriteLine($"{text}  ->  [{string.Join(",", ids)}]  ->  {back}");
            Assert.NotEmpty(ids);
            Assert.Equal(text, back); // encode -> decode is lossless for ordinary text
        }

        // Known reference ids from the upstream HF `tokenizers` library (byte-level BPE, no specials).
        Assert.Equal(new[] { 785, 6722, 315, 9625, 374 }, tok.Encode("The capital of France is"));
        // Digits split one-by-one (the Qwen pre-tokenizer pattern), unlike GPT-2's digit runs.
        Assert.Equal(new[] { 21927, 11, 1879, 0, 220, 16, 17, 18, 19, 20 }, tok.Encode(" Hello, world! 12345"));
    }

    [Fact]
    public void HfTokenizer_Special_Tokens_Are_Single_Units()
    {
        if (!TryResolveTokenizer(out string tokPath))
        {
            _out.WriteLine("Qwen tokenizer.json not found; skipping.");
            return;
        }

        BpeTokenizer tok = HfTokenizerJson.FromFile(tokPath);

        // <|im_start|> (151644) is an added/special token: it must encode as exactly one id...
        int[] ids = tok.Encode("<|im_start|>user");
        Assert.Equal(new[] { 151644, 872 }, ids);

        // ...be dropped on a normal decode...
        Assert.Equal("user", tok.Decode(ids));
        // ...and be re-emitted verbatim when specials are kept.
        Assert.Equal("<|im_start|>user", tok.Decode(ids, skipSpecial: false));
    }

    // ---- full end-to-end: encode -> generate -> decode to real text ----

    [Fact]
    public void Qwen05b_Generates_Real_Decoded_Text()
    {
        if (!RealModelAssets.TryResolveOrDownload(ModelRel, HubSpec, out string modelPath, log: _out.WriteLine) ||
            !TryResolveTokenizer(out string tokPath))
        {
            _out.WriteLine("Qwen model or tokenizer not found; skipping.");
            return;
        }

        BpeTokenizer tok = HfTokenizerJson.FromFile(tokPath);
        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        _out.WriteLine($"Loaded Qwen-0.5B q4 ({g.Nodes.Count} nodes); tokenizer VocabSize={tok.VocabSize}.");

        const string prompt = "The capital of France is";
        int[] promptIds = tok.Encode(prompt);
        _out.WriteLine($"Prompt: \"{prompt}\"  ->  [{string.Join(",", promptIds)}]");
        Assert.NotEmpty(promptIds);

        const int maxNewTokens = 16;
        List<int> generated = Generate(g, promptIds, maxNewTokens);

        string text = tok.Decode(generated);
        _out.WriteLine($"Generated ids: [{string.Join(",", generated)}]");
        _out.WriteLine($"Generated text: \"{text}\"");

        // Non-empty, contains real letters (decoded text, not just punctuation/specials).
        Assert.NotEmpty(generated);
        Assert.False(string.IsNullOrWhiteSpace(text));
        Assert.Contains(text, c => char.IsLetter(c));

        // Greedy decoding is deterministic: a second run yields identical ids and text.
        List<int> generated2 = Generate(g, promptIds, maxNewTokens);
        Assert.Equal(generated, generated2);
        Assert.Equal(text, tok.Decode(generated2));
    }

    /// <summary>
    /// Self-contained greedy loop: prefill the prompt, then repeatedly argmax the last-position logits,
    /// append, and feed the previous step's KV cache (<c>present.*</c> → <c>past_key_values.*</c>) plus a
    /// single new token. Returns only the newly generated ids (prompt excluded). Stops at an EOS id.
    /// </summary>
    private List<int> Generate(ModelGraph g, int[] promptIds, int maxNewTokens)
    {
        using IExecutionEngine eng = new ManagedCpuEngine(g);

        var generated = new List<int>(maxNewTokens);
        var past = new Dictionary<string, NamedTensor>(); // empty -> length-0 KV on the first (prefill) step
        int pastLen = 0;
        long[] stepIds = promptIds.Select(i => (long)i).ToArray();

        for (int step = 0; step < maxNewTokens; step++)
        {
            int curSeq = stepIds.Length;
            Dictionary<string, NamedTensor> feeds = BuildFeeds(g.Inputs, stepIds, pastLen, past);

            IReadOnlyDictionary<string, NamedTensor> outp = eng.Run(feeds);

            Tensor<float> logits = outp["logits"].Data;
            int[] dims = logits.Shape.Dimensions.ToArray();
            int vocab = dims[^1], seq = dims[^2];
            float[] all = logits.Buffer.ToArray();

            int baseOff = (seq - 1) * vocab; // logits for the last position
            int best = 0; float bestVal = all[baseOff];
            for (int v = 1; v < vocab; v++)
                if (all[baseOff + v] > bestVal) { bestVal = all[baseOff + v]; best = v; }

            generated.Add(best);
            if (EosIds.Contains(best)) break;

            // Carry forward the produced KV cache as the next step's past.
            past = CapturePresent(outp);
            pastLen += curSeq;
            stepIds = new[] { (long)best }; // decode one token at a time
        }

        return generated;
    }

    private static Dictionary<string, NamedTensor> BuildFeeds(
        IReadOnlyList<string> inputs, long[] tokenIds, int pastLen, Dictionary<string, NamedTensor> past)
    {
        int seq = tokenIds.Length;
        int total = pastLen + seq;
        var feeds = new Dictionary<string, NamedTensor>();
        foreach (string name in inputs)
        {
            if (name == "input_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq), tokenIds));
            else if (name == "attention_mask")
                // Mask covers the whole context (past + current).
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, total),
                    Enumerable.Repeat(1L, total).ToArray()));
            else if (name == "position_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq),
                    Enumerable.Range(pastLen, seq).Select(i => (long)i).ToArray()));
            else if (name.StartsWith("past_key_values", StringComparison.Ordinal))
            {
                if (past.TryGetValue(name, out NamedTensor? carried))
                    feeds[name] = carried;
                else
                    feeds[name] = new NamedTensor(name, new Tensor<float>(new TensorShape(1, KvHeads, pastLen, HeadDim)));
            }
            else
                throw new InvalidOperationException($"Unexpected Qwen input '{name}'.");
        }
        return feeds;
    }

    /// <summary>Maps the model's <c>present.N.{key,value}</c> outputs to the next step's
    /// <c>past_key_values.N.{key,value}</c> feed names.</summary>
    private static Dictionary<string, NamedTensor> CapturePresent(IReadOnlyDictionary<string, NamedTensor> outp)
    {
        var past = new Dictionary<string, NamedTensor>();
        foreach (KeyValuePair<string, NamedTensor> kv in outp)
        {
            if (!kv.Key.StartsWith("present", StringComparison.Ordinal)) continue;
            string pastName = "past_key_values" + kv.Key.Substring("present".Length);
            past[pastName] = new NamedTensor(pastName, kv.Value.Tensor);
        }
        return past;
    }
}
