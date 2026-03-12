using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Tada;

public class TensorInfo
{
    [JsonPropertyName("dtype")]
    public string Dtype { get; set; } = "";

    [JsonPropertyName("shape")]
    public int[] Shape { get; set; } = Array.Empty<int>();

    [JsonPropertyName("data_offsets")]
    public long[] DataOffsets { get; set; } = Array.Empty<long>();
}

public class SafetensorsReader : IDisposable
{
    private readonly FileStream _fs;
    private readonly Dictionary<string, TensorInfo> _metadata;
    private readonly long _dataStartOffset;

    public SafetensorsReader(string filePath)
    {
        _fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);

        byte[] lengthBytes = new byte[8];
        _fs.ReadExactly(lengthBytes);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(lengthBytes);

        byte[] headerBytes = new byte[headerLength];
        _fs.ReadExactly(headerBytes);

        string headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);
        _metadata = JsonSerializer.Deserialize<Dictionary<string, TensorInfo>>(headerJson) ?? new Dictionary<string, TensorInfo>();

        _dataStartOffset = 8 + headerLength;
    }

    public bool HasTensor(string name)
    {
        return _metadata.ContainsKey(name);
    }

    public int[] GetTensorShape(string name)
    {
        return _metadata[name].Shape;
    }

    public float[] GetTensor(string name)
    {
        if (!_metadata.TryGetValue(name, out var info))
        {
            throw new KeyNotFoundException($"Tensor '{name}' not found.");
        }

        if (info.Dtype != "F32")
        {
            throw new NotSupportedException($"Only F32 tensors are supported. Found {info.Dtype} for {name}");
        }

        long start = info.DataOffsets[0];
        long end = info.DataOffsets[1];
        long byteLength = end - start;

        int elementCount = (int)(byteLength / 4);
        float[] result = new float[elementCount];

        _fs.Seek(_dataStartOffset + start, SeekOrigin.Begin);

        Span<byte> byteSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<float, byte>(result.AsSpan());
        int read = _fs.Read(byteSpan);
        if (read != byteLength)
        {
            throw new IOException($"Failed to read full tensor data for {name}");
        }

        if (!BitConverter.IsLittleEndian)
        {
            throw new NotSupportedException("Big-endian systems not supported.");
        }

        return result;
    }

    public void Dispose()
    {
        _fs.Dispose();
    }
}
