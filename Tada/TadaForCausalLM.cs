using System;

namespace Tada;

public class TadaForCausalLM
{
    private readonly TadaConfig _config;
    public int NumTimeBits { get; }

    public TadaForCausalLM(TadaConfig config)
    {
        _config = config;
        NumTimeBits = (int)Math.Ceiling(Math.Log2(config.NumTimeClasses));
    }

    public float[] Generate(EncoderOutput prompt, string text, InferenceOptions options)
    {
        // Stub: Simply copy prompt audio and text as our "generation"
        return prompt.Audio;
    }
}
