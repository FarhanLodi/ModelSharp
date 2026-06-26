using System.Collections.Generic;

namespace ModelSharp.Text;

/// <summary>
/// GPT-2 byte-level mapping (the <c>bytes_to_unicode</c> table). Maps every one of the 256 byte
/// values to a distinct printable Unicode code point so that raw UTF-8 bytes can be carried through
/// a text-based BPE merge process without control / whitespace characters disrupting the regex
/// pre-tokenization. The mapping is a bijection over 0..255 and is fully reversible, which lets the
/// tokenizer round-trip arbitrary UTF-8 (including multibyte / emoji) text.
/// </summary>
public static class ByteLevel
{
    private static readonly char[] _byteToChar;
    private static readonly Dictionary<char, byte> _charToByte;

    static ByteLevel()
    {
        // The bytes that already correspond to printable, non-whitespace Unicode characters map to
        // themselves; everything else is shifted up into the 256+ range. Ranges mirror GPT-2:
        //   '!'..'~'  (33..126), '¡'..'¬' (0xA1..0xAC), '®'..'ÿ' (0xAE..0xFF).
        var printable = new bool[256];
        MarkRange(printable, '!', '~');
        MarkRange(printable, 0xA1, 0xAC);
        MarkRange(printable, 0xAE, 0xFF);

        _byteToChar = new char[256];
        for (int b = 0; b < 256; b++)
            if (printable[b]) _byteToChar[b] = (char)b;

        // Remaining (non-printable) bytes are assigned, in ascending byte order, to the code points
        // 256, 257, ... — guaranteeing distinct, BMP, non-whitespace characters (byte 0x20 → 'Ġ').
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!printable[b]) _byteToChar[b] = (char)(256 + n++);

        _charToByte = new Dictionary<char, byte>(256);
        for (int b = 0; b < 256; b++)
            _charToByte[_byteToChar[b]] = (byte)b;
    }

    private static void MarkRange(bool[] flags, int loInclusive, int hiInclusive)
    {
        for (int i = loInclusive; i <= hiInclusive; i++) flags[i] = true;
    }

    /// <summary>The full byte → character table; index is the byte value, length is 256.</summary>
    public static IReadOnlyList<char> ByteEncoderTable => _byteToChar;

    /// <summary>Maps a byte to its GPT-2 byte-level character.</summary>
    public static char EncodeByte(byte b) => _byteToChar[b];

    /// <summary>Maps a byte-level character back to its original byte; false if <paramref name="c"/> is not a byte-level character.</summary>
    public static bool TryDecodeChar(char c, out byte b) => _charToByte.TryGetValue(c, out b);
}
