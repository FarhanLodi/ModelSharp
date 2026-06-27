using System;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// Verifies downloaded files against expected <see cref="RepoFile"/> metadata. Hashing is streamed
/// through <see cref="SHA256"/> so large model files are never loaded into memory.
/// </summary>
public static class IntegrityVerifier
{
    /// <summary>
    /// Computes the SHA-256 of the file at <paramref name="path"/>, streaming its bytes through the hash
    /// algorithm (the file is never fully buffered in memory), and returns the lowercase hex digest.
    /// </summary>
    public static async Task<string> Sha256Async(string path, CancellationToken ct = default)
    {
        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 1 << 20,
            options: FileOptions.Asynchronous | FileOptions.SequentialScan);

        var hash = await SHA256.HashDataAsync(stream, ct).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }

    /// <summary>
    /// Verifies the file at <paramref name="path"/> against <paramref name="expected"/>. Prefers a
    /// SHA-256 comparison when one is supplied, then falls back to a size comparison, then to mere
    /// existence. A missing file yields <c>false</c>.
    /// </summary>
    public static async Task<bool> VerifyAsync(string path, RepoFile expected, CancellationToken ct = default)
    {
        if (!File.Exists(path))
            return false;

        if (!string.IsNullOrEmpty(expected.Sha256))
        {
            var actual = await Sha256Async(path, ct).ConfigureAwait(false);
            return string.Equals(actual, expected.Sha256, StringComparison.OrdinalIgnoreCase);
        }

        if (expected.Size is { } size)
        {
            var actualLength = new FileInfo(path).Length;
            return actualLength == size;
        }

        return true;
    }

    /// <summary>
    /// Verifies the file and throws <see cref="HubException"/> (reporting expected vs actual) when it
    /// fails to match the supplied metadata.
    /// </summary>
    public static async Task EnsureAsync(string path, RepoFile expected, CancellationToken ct = default)
    {
        if (!File.Exists(path))
            throw new HubException($"Integrity check failed for '{path}': file does not exist.");

        if (!string.IsNullOrEmpty(expected.Sha256))
        {
            var actual = await Sha256Async(path, ct).ConfigureAwait(false);
            if (!string.Equals(actual, expected.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new HubException(
                    $"SHA-256 mismatch for '{path}': expected {expected.Sha256.ToLowerInvariant()}, actual {actual}.");
            }

            return;
        }

        if (expected.Size is { } size)
        {
            var actualLength = new FileInfo(path).Length;
            if (actualLength != size)
            {
                throw new HubException(
                    $"Size mismatch for '{path}': expected {size} bytes, actual {actualLength} bytes.");
            }

            return;
        }

        // No verifiable metadata: existence (checked above) is sufficient.
    }
}
