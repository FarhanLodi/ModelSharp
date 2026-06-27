using System.Collections.Generic;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.ControlFlow;

/// <summary>
/// ONNX <c>Loop</c>: a generic loop with an optional trip-count bound <c>M</c>, an optional
/// boolean termination condition <c>cond</c>, and N loop-carried dependencies.
/// <para>
/// Node inputs:  <c>M?, cond?, v_init_1 .. v_init_N</c><br/>
/// Node outputs: <c>v_final_1 .. v_final_N, scan_out_1 .. scan_out_K</c>
/// </para>
/// The <c>body</c> subgraph signature is
/// <c>(iter_num i64, cond_in bool, v_1..v_N) -> (cond_out bool, v_1..v_N, scan_1..scan_K)</c>.
/// Each iteration: feed the counter, the running condition, and the carried values; read back
/// the next condition, the next carried values, and one slice of each scan output. Scan outputs
/// are stacked along a new leading axis (iteration count). Termination is when the trip count is
/// reached or <c>cond_out</c> is false (whichever comes first); an absent bound / condition is
/// treated as "no limit" / "always true" respectively.
/// </summary>
public sealed class LoopKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Loop";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        ModelGraph body = CfAttr.Graph(node, "body");

        // Optional M (input 0) and cond (input 1). Empty input name => omitted.
        bool hasMax = node.Inputs.Count > 0 && node.Inputs[0].Length != 0;
        long maxTrips = hasMax ? CfTensorOps.ReadScalarInt(ctx.GetTensor(node.Inputs[0])) : long.MaxValue;

        bool hasCond = node.Inputs.Count > 1 && node.Inputs[1].Length != 0;
        bool keepGoing = !hasCond || IfKernel.ReadBoolScalar(ctx.GetTensor(node.Inputs[1]), node);

        // Loop-carried initial values: node inputs [2..].
        int carryCount = node.Inputs.Count - 2;
        var carried = new Tensor[carryCount];
        for (int i = 0; i < carryCount; i++)
            carried[i] = ctx.GetTensor(node.Inputs[2 + i]);

        // body outputs: cond_out, v_1..v_N, scan_1..scan_K
        int scanCount = body.Outputs.Count - 1 - carryCount;
        if (scanCount < 0)
            throw new ModelSharpException(
                $"Loop node '{node.Name}': body has {body.Outputs.Count} outputs but "
                + $"{carryCount} carried + 1 cond expected.");
        var scanAccum = new List<Tensor>[scanCount];
        for (int k = 0; k < scanCount; k++) scanAccum[k] = new List<Tensor>();

        long iter = 0;
        while (iter < maxTrips && keepGoing)
        {
            var feeds = new Dictionary<string, Tensor>();
            // body inputs: iter_num (i64), cond_in (bool), v_1..v_N
            feeds[body.Inputs[0]] = CfTensorOps.Int64Scalar(iter);
            feeds[body.Inputs[1]] = CfTensorOps.BoolScalar(keepGoing);
            for (int i = 0; i < carryCount; i++)
                feeds[body.Inputs[2 + i]] = carried[i];

            IReadOnlyDictionary<string, Tensor> outs = ctx.RunSubgraph(body, feeds);

            // body outputs: cond_out, v_1..v_N, scan_1..scan_K
            keepGoing = IfKernel.ReadBoolScalar(outs[body.Outputs[0]], node);
            for (int i = 0; i < carryCount; i++)
                carried[i] = outs[body.Outputs[1 + i]];
            for (int k = 0; k < scanCount; k++)
                scanAccum[k].Add(outs[body.Outputs[1 + carryCount + k]]);

            iter++;
        }

        // Node outputs: v_final_1..v_final_N, scan_out_1..scan_out_K
        for (int i = 0; i < carryCount && i < node.Outputs.Count; i++)
            ctx.Set(node.Outputs[i], carried[i]);
        for (int k = 0; k < scanCount && carryCount + k < node.Outputs.Count; k++)
            ctx.Set(node.Outputs[carryCount + k], CfTensorOps.Stack(scanAccum[k], axis: 0));
    }
}
