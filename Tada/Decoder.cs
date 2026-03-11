using System;

namespace Tada;

public class DecoderConfig
{
    public int EmbedDim { get; set; } = 512;
    public int HiddenDim { get; set; } = 1024;
    public int NumAttnLayers { get; set; } = 6;
    public int NumAttnHeads { get; set; } = 8;
    public int AttnDimFeedforward { get; set; } = 4096;
}

public class Decoder
{
    private readonly DecoderConfig _config;

    public Decoder(DecoderConfig config)
    {
        _config = config;
    }

    public float[] Decode(ReadOnlySpan<float> input)
    {
        // Stub implementation, returns unmodified input array
        return input.ToArray();
    }
}
