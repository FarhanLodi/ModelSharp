using System;
using System.Text;

namespace ModelSharp.Onnx;

/// <summary>
/// Minimal protobuf wire-format reader over a byte buffer. Reads only what the ONNX
/// subset needs; unknown fields are skipped. No Google.Protobuf dependency — this keeps
/// the loader pure-managed and dependency-free.
/// </summary>
internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _buf;
    private int _pos;

    public ProtoReader(ReadOnlySpan<byte> buf)
    {
        _buf = buf;
        _pos = 0;
    }

    public readonly bool Eof => _pos >= _buf.Length;

    /// <summary>Reads a field tag. Returns false at EOF; outputs field number + wire type.</summary>
    public bool TryReadTag(out int fieldNumber, out int wireType)
    {
        if (Eof) { fieldNumber = 0; wireType = 0; return false; }
        ulong tag = ReadVarint();
        fieldNumber = (int)(tag >> 3);
        wireType = (int)(tag & 0x7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            if (_pos >= _buf.Length) throw new FormatException("Truncated varint in protobuf stream.");
            byte b = _buf[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
            if (shift > 63) throw new FormatException("Varint exceeds 64 bits.");
        }
        return result;
    }

    public long ReadInt64() => (long)ReadVarint();
    public int ReadInt32() => (int)(long)ReadVarint();

    public uint ReadFixed32()
    {
        if (_pos + 4 > _buf.Length) throw new FormatException("Truncated fixed32.");
        uint v = (uint)(_buf[_pos] | (_buf[_pos + 1] << 8) | (_buf[_pos + 2] << 16) | (_buf[_pos + 3] << 24));
        _pos += 4;
        return v;
    }

    public float ReadFloat() => BitConverter.UInt32BitsToSingle(ReadFixed32());

    public ulong ReadFixed64()
    {
        if (_pos + 8 > _buf.Length) throw new FormatException("Truncated fixed64.");
        ulong v = 0;
        for (int i = 0; i < 8; i++) v |= (ulong)_buf[_pos + i] << (8 * i);
        _pos += 8;
        return v;
    }

    public double ReadDouble() => BitConverter.UInt64BitsToDouble(ReadFixed64());

    /// <summary>Reads a length-delimited field as a sub-span (no copy).</summary>
    public ReadOnlySpan<byte> ReadLengthDelimited()
    {
        int len = (int)ReadVarint();
        if (_pos + len > _buf.Length) throw new FormatException("Length-delimited field exceeds buffer.");
        ReadOnlySpan<byte> slice = _buf.Slice(_pos, len);
        _pos += len;
        return slice;
    }

    public string ReadString() => Encoding.UTF8.GetString(ReadLengthDelimited());

    /// <summary>Skips a field of the given wire type.</summary>
    public void SkipField(int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarint(); break;
            case 1: ReadFixed64(); break;
            case 2: ReadLengthDelimited(); break;
            case 5: ReadFixed32(); break;
            default: throw new FormatException($"Unsupported protobuf wire type {wireType}.");
        }
    }
}
