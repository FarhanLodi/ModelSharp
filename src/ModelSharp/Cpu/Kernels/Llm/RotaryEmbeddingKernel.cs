using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// Rotary position embedding (RoPE), the ONNXRuntime / HuggingFace contrib op
/// <c>RotaryEmbedding</c>. Rotates the leading <c>rotary_embedding_dim</c> channels of
/// each attention head using per-position sine/cosine tables; channels beyond the
/// rotary span pass through unchanged.
/// </summary>
/// <remarks>
/// <para>Input layout: either <c>[batch, seq, hidden]</c> (3-D) — hidden is reshaped to
/// <c>num_heads · head_size</c> — or <c>[batch, num_heads, seq, head_size]</c> (4-D).</para>
/// <para><c>cos_cache</c> and <c>sin_cache</c> are <c>[max_position, rotary_dim/2]</c>; for a
/// token at position <c>p</c> the half-vector <c>cos[p, :]</c> / <c>sin[p, :]</c> is gathered
/// (looked up via <c>position_ids</c> when supplied, else the token's sequence index).</para>
/// <para>Two layouts of the rotary span are supported, matching the reference op:</para>
/// <list type="bullet">
/// <item><description><b>half-split (GPT-NeoX, <c>interleaved = 0</c>)</b>: pairs are
/// <c>(x[i], x[i + d/2])</c> for <c>i ∈ [0, d/2)</c>, rotated as
/// <c>x'[i] = x[i]·cos[i] − x[i+d/2]·sin[i]</c>,
/// <c>x'[i+d/2] = x[i+d/2]·cos[i] + x[i]·sin[i]</c>.</description></item>
/// <item><description><b>interleaved (GPT-J, <c>interleaved = 1</c>)</b>: pairs are
/// <c>(x[2j], x[2j+1])</c>, rotated as
/// <c>x'[2j] = x[2j]·cos[j] − x[2j+1]·sin[j]</c>,
/// <c>x'[2j+1] = x[2j+1]·cos[j] + x[2j]·sin[j]</c>.</description></item>
/// </list>
/// </remarks>
public sealed class RotaryEmbeddingKernel : IKernel
{
    public string OpType => "RotaryEmbedding";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> input = ctx.Get(node.Inputs[0]);
        long[] positionIds = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        Tensor<float> cosCache = ctx.Get(node.Inputs[2]);
        Tensor<float> sinCache = ctx.Get(node.Inputs[3]);

        bool interleaved = Attr.Int(node, "interleaved", 0) != 0;
        int numHeads = (int)Attr.Int(node, "num_heads", 0);
        int rotaryDim = (int)Attr.Int(node, "rotary_embedding_dim", 0);

        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        int batch, seq, heads, headSize;
        if (dims.Length == 4)
        {
            batch = dims[0]; heads = dims[1]; seq = dims[2]; headSize = dims[3];
        }
        else if (dims.Length == 3)
        {
            batch = dims[0]; seq = dims[1];
            heads = numHeads > 0 ? numHeads : 1;
            headSize = dims[2] / heads;
        }
        else
        {
            throw new InvalidOperationException(
                $"RotaryEmbedding expects a 3-D [B,S,H] or 4-D [B,N,S,D] input; got rank {dims.Length}.");
        }

        // cos/sin cache hold rotary_dim/2 columns per position.
        int half = cosCache.Shape.Dimensions[cosCache.Shape.Rank - 1];
        int rotary = rotaryDim > 0 ? rotaryDim : 2 * half;
        if (rotary > headSize) rotary = headSize;
        int rotHalf = rotary / 2;

        var y = new Tensor<float>(input.Shape);
        Span<float> xs = input.Span, ys = y.Span, cos = cosCache.Span, sin = sinCache.Span;
        xs.CopyTo(ys);

        // Stride to step from one (batch, head, token) row to the next inside `xs`.
        // For 4-D [B,N,S,D] the token axis is innermost-but-one; for 3-D [B,S,H] heads
        // are packed inside hidden so a head's row is contiguous of length headSize.
        for (int bch = 0; bch < batch; bch++)
        {
            for (int s = 0; s < seq; s++)
            {
                long pos = positionIds.Length == 1
                    ? positionIds[0] + s
                    : positionIds[Math.Min(bch * seq + s, positionIds.Length - 1)];
                int cacheBase = (int)pos * half;

                for (int h = 0; h < heads; h++)
                {
                    int rowBase = dims.Length == 4
                        ? ((bch * heads + h) * seq + s) * headSize
                        : ((bch * seq + s) * heads + h) * headSize;

                    for (int j = 0; j < rotHalf; j++)
                    {
                        float c = cos[cacheBase + j];
                        float sn = sin[cacheBase + j];
                        int i0, i1;
                        if (interleaved) { i0 = rowBase + 2 * j; i1 = i0 + 1; }
                        else { i0 = rowBase + j; i1 = rowBase + j + rotHalf; }

                        float a = xs[i0];
                        float b = xs[i1];
                        ys[i0] = a * c - b * sn;
                        ys[i1] = b * c + a * sn;
                    }
                }
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
