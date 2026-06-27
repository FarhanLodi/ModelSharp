using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Sequence;

/// <summary>
/// A non-tensor runtime value produced/consumed by the ONNX <c>Sequence*</c> and
/// <c>Optional*</c> op families. These values live <b>only</b> on the intra-graph wire
/// between nodes — graph inputs and outputs remain plain tensors, so the public
/// <see cref="ModelSharp.Engine.IExecutionEngine.Run"/> contract is unaffected.
///
/// <para>
/// Two kinds are modeled:
/// <list type="bullet">
/// <item><b>Sequence</b> — an ordered, homogeneous list of tensors (ONNX <c>seq(T)</c>).</item>
/// <item><b>Optional</b> — a present-or-absent wrapper (ONNX <c>optional(T)</c>); when present
/// it wraps either a tensor or a sequence.</item>
/// </list>
/// </para>
/// </summary>
public sealed class SeqValue
{
    /// <summary>The element kind this value carries.</summary>
    public enum ValueKind
    {
        /// <summary>An ordered list of tensors (a tensor sequence).</summary>
        Sequence,

        /// <summary>An optional present value wrapping a tensor.</summary>
        OptionalTensor,

        /// <summary>An optional present value wrapping a sequence.</summary>
        OptionalSequence,

        /// <summary>An optional absent (none) value.</summary>
        OptionalNone,
    }

    /// <summary>Which kind of non-tensor value this is.</summary>
    public ValueKind Kind { get; }

    /// <summary>
    /// The backing tensor list. For <see cref="ValueKind.Sequence"/> this is the sequence's
    /// elements; for <see cref="ValueKind.OptionalSequence"/> the wrapped sequence's elements;
    /// for <see cref="ValueKind.OptionalTensor"/> a single-element list; for
    /// <see cref="ValueKind.OptionalNone"/> empty.
    /// </summary>
    public IReadOnlyList<Tensor> Tensors { get; }

    private SeqValue(ValueKind kind, IReadOnlyList<Tensor> tensors)
    {
        Kind = kind;
        Tensors = tensors;
    }

    /// <summary>Builds a tensor-sequence value from an ordered tensor list (the list is taken as-is).</summary>
    public static SeqValue Sequence(IReadOnlyList<Tensor> tensors) =>
        new(ValueKind.Sequence, tensors);

    /// <summary>Builds a present optional wrapping a single tensor.</summary>
    public static SeqValue SomeTensor(Tensor t) =>
        new(ValueKind.OptionalTensor, new[] { t });

    /// <summary>Builds a present optional wrapping a sequence.</summary>
    public static SeqValue SomeSequence(SeqValue seq) =>
        new(ValueKind.OptionalSequence, seq.Tensors);

    /// <summary>The canonical absent optional ("none").</summary>
    public static SeqValue None { get; } = new(ValueKind.OptionalNone, System.Array.Empty<Tensor>());

    /// <summary>True when this is an optional kind (present or absent).</summary>
    public bool IsOptional =>
        Kind is ValueKind.OptionalTensor or ValueKind.OptionalSequence or ValueKind.OptionalNone;

    /// <summary>True when an optional is present (a tensor or a sequence), false for none.</summary>
    public bool HasElement => Kind is ValueKind.OptionalTensor or ValueKind.OptionalSequence;

    /// <summary>Number of tensors in a sequence (or wrapped sequence).</summary>
    public int Count => Tensors.Count;
}
