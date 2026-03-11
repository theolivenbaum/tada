using System;

namespace Tada;

public class EncoderConfig
{
    public int HiddenDim { get; set; } = 1024;
    public int EmbedDim { get; set; } = 512;
    public int[] Strides { get; set; } = new[] { 6, 5, 4, 4 };
    public int NumAttnLayers { get; set; } = 6;
    public int NumAttnHeads { get; set; } = 8;
}

public class EncoderOutput
{
    // Keeping it simple for the initial port
    public float[] Audio { get; set; } = Array.Empty<float>();
    public float[] AudioLen { get; set; } = Array.Empty<float>();
    public string[] Text { get; set; } = Array.Empty<string>();
}

public class Encoder
{
    private readonly EncoderConfig _config;

    public Encoder(EncoderConfig config)
    {
        _config = config;
    }

    public EncoderOutput Encode(ReadOnlySpan<float> audioSamples, string[] text)
    {
        // Stub implementation
        return new EncoderOutput
        {
            Audio = audioSamples.ToArray(),
            Text = text
        };
    }
}
