using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.ControlFlow;

/// <summary>
/// ONNX <c>Scan</c>: iterates a <c>body</c> subgraph over one or more scan inputs, threading
/// state variables across iterations and stacking per-iteration scan outputs.
/// <para>
/// Node inputs:  <c>init_state_1..init_state_M, scan_in_1..scan_in_K</c>
/// (the last <c>num_scan_inputs</c> inputs are the scan inputs).<br/>
/// Node outputs: <c>final_state_1..final_state_M, scan_out_1..scan_out_K</c>.
/// </para>
/// Body signature: <c>(state_1..state_M, slice_1..slice_K) -> (state_1..state_M, out_1..out_K)</c>.
/// Each iteration <c>t</c> feeds slice <c>t</c> of each scan input (taken along its
/// <c>scan_input_axes</c>, default 0) and the running states; reads back the next states and one
/// slice of each scan output, which are stacked along <c>scan_output_axes</c> (default 0).
/// <c>scan_input_directions</c> / <c>scan_output_directions</c> (0 = forward, 1 = reverse) are honored.
/// </summary>
public sealed class ScanKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Scan";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        ModelGraph body = CfAttr.Graph(node, "body");

        int numScan = (int)Attr.Int(node, "num_scan_inputs", 0);
        if (numScan <= 0)
            throw new ModelSharpException(
                $"Scan node '{node.Name}': num_scan_inputs must be >= 1, got {numScan}.");

        int stateCount = node.Inputs.Count - numScan;
        if (stateCount < 0)
            throw new ModelSharpException(
                $"Scan node '{node.Name}': num_scan_inputs {numScan} exceeds input count {node.Inputs.Count}.");

        int scanOutCount = body.Outputs.Count - stateCount;

        int[] inAxes = Attr.Ints(node, "scan_input_axes", DefaultAxes(numScan));
        int[] outAxes = Attr.Ints(node, "scan_output_axes", DefaultAxes(scanOutCount));
        int[] inDirs = Attr.Ints(node, "scan_input_directions", DefaultAxes(numScan));
        int[] outDirs = Attr.Ints(node, "scan_output_directions", DefaultAxes(scanOutCount));

        // Running state values (node inputs [0..stateCount)).
        var state = new Tensor[stateCount];
        for (int i = 0; i < stateCount; i++) state[i] = ctx.GetTensor(node.Inputs[i]);

        // Scan inputs (node inputs [stateCount..]).
        var scanIn = new Tensor[numScan];
        for (int k = 0; k < numScan; k++) scanIn[k] = ctx.GetTensor(node.Inputs[stateCount + k]);

        // Sequence length = extent of each scan input along its scan axis (must agree).
        int seqLen = -1;
        for (int k = 0; k < numScan; k++)
        {
            int ax = NormAxis(inAxes[k], scanIn[k].Shape.Rank);
            int len = scanIn[k].Shape[ax];
            if (seqLen < 0) seqLen = len;
            else if (seqLen != len)
                throw new ModelSharpException(
                    $"Scan node '{node.Name}': scan inputs disagree on sequence length ({seqLen} vs {len}).");
        }
        if (seqLen < 0) seqLen = 0;

        var scanAccum = new List<Tensor>[scanOutCount];
        for (int k = 0; k < scanOutCount; k++) scanAccum[k] = new List<Tensor>();

        for (int t = 0; t < seqLen; t++)
        {
            var feeds = new Dictionary<string, Tensor>();
            // body inputs: state_1..state_M, slice_1..slice_K
            for (int i = 0; i < stateCount; i++)
                feeds[body.Inputs[i]] = state[i];
            for (int k = 0; k < numScan; k++)
            {
                int ax = NormAxis(inAxes[k], scanIn[k].Shape.Rank);
                int idx = inDirs[k] == 1 ? seqLen - 1 - t : t;
                feeds[body.Inputs[stateCount + k]] = CfTensorOps.SliceAlongAxis(scanIn[k], ax, idx);
            }

            IReadOnlyDictionary<string, Tensor> outs = ctx.RunSubgraph(body, feeds);

            // body outputs: state_1..state_M, out_1..out_K
            for (int i = 0; i < stateCount; i++)
                state[i] = outs[body.Outputs[i]];
            for (int k = 0; k < scanOutCount; k++)
                scanAccum[k].Add(outs[body.Outputs[stateCount + k]]);
        }

        // Node outputs: final_state_1..final_state_M, scan_out_1..scan_out_K
        for (int i = 0; i < stateCount && i < node.Outputs.Count; i++)
            ctx.Set(node.Outputs[i], state[i]);
        for (int k = 0; k < scanOutCount && stateCount + k < node.Outputs.Count; k++)
        {
            // Reverse the accumulated slices for reverse-direction outputs before stacking.
            if (outDirs[k] == 1) scanAccum[k].Reverse();
            ctx.Set(node.Outputs[stateCount + k], CfTensorOps.Stack(scanAccum[k], outAxes[k]));
        }
    }

    private static int NormAxis(int axis, int rank) => axis < 0 ? axis + rank : axis;

    private static int[] DefaultAxes(int count) => new int[count]; // all zeros
}
