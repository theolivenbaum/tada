using System;

namespace Tada;

public class TadaForCausalLM
{
    private readonly TadaConfig _config;
    public int NumTimeBits { get; }

    private readonly VibeVoiceDiffusionHead _predictionHead;
    private readonly float[] _bottleneckProjWeight;

    public TadaForCausalLM(TadaConfig config)
    {
        _config = config;
        NumTimeBits = (int)Math.Ceiling(Math.Log2(config.NumTimeClasses));

        int timeDim = 2 * NumTimeBits;

        // Configuration defaults
        int headLayers = 4;
        float headFfnRatio = 3.0f;
        int bottleneckDim = config.HiddenSize; // Assuming no bottleneck for simplicity

        _predictionHead = new VibeVoiceDiffusionHead(
            bottleneckDim,
            config.AcousticDim + timeDim,
            headLayers,
            headFfnRatio
        );

        _bottleneckProjWeight = new float[bottleneckDim * config.HiddenSize];
    }

    private void ComputeVelocity(
        ReadOnlySpan<float> speechInput,
        float t,
        ReadOnlySpan<float> condInput,
        ReadOnlySpan<float> negCondInput,
        float acousticCfg,
        float durationCfg,
        Span<float> velocity,
        int batchSize)
    {
        int totalDim = _config.AcousticDim + 2 * NumTimeBits;

        Span<float> tSpan = stackalloc float[batchSize];
        tSpan.Fill(t);

        if (acousticCfg != 1.0f)
        {
            // Duplicate speech input
            Span<float> speechCombined = stackalloc float[batchSize * 2 * totalDim];
            speechInput.CopyTo(speechCombined);
            speechInput.CopyTo(speechCombined.Slice(batchSize * totalDim));

            Span<float> tCombined = stackalloc float[batchSize * 2];
            tCombined.Fill(t);

            // Duplicate condition input
            Span<float> condCombined = stackalloc float[batchSize * 2 * _config.HiddenSize];
            condInput.CopyTo(condCombined);
            negCondInput.CopyTo(condCombined.Slice(batchSize * _config.HiddenSize));

            // Project condition if needed
            Span<float> bottleneckCond = stackalloc float[batchSize * 2 * _config.HiddenSize];
            TensorOperations.Linear(condCombined, bottleneckCond, _config.HiddenSize, _config.HiddenSize, batchSize * 2, _bottleneckProjWeight, ReadOnlySpan<float>.Empty);

            Span<float> velocityCombined = stackalloc float[batchSize * 2 * totalDim];
            _predictionHead.Forward(speechCombined, tCombined, bottleneckCond, velocityCombined, batchSize * 2);

            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < totalDim; d++)
                {
                    int posIdx = b * totalDim + d;
                    int negIdx = (batchSize + b) * totalDim + d;
                    float vPos = velocityCombined[posIdx];
                    float vNeg = velocityCombined[negIdx];

                    float cfg = d < _config.AcousticDim ? acousticCfg : durationCfg;
                    velocity[posIdx] = vNeg + cfg * (vPos - vNeg);
                }
            }
        }
        else
        {
            Span<float> bottleneckCond = stackalloc float[batchSize * _config.HiddenSize];
            TensorOperations.Linear(condInput, bottleneckCond, _config.HiddenSize, _config.HiddenSize, batchSize, _bottleneckProjWeight, ReadOnlySpan<float>.Empty);

            _predictionHead.Forward(speechInput, tSpan, bottleneckCond, velocity, batchSize);
        }
    }

    private static float ScheduledCfg(float baseScale, float t, string schedule)
    {
        if (schedule == "constant" || baseScale == 1.0f)
            return baseScale;
        if (schedule == "linear")
            return 1.0f + (baseScale - 1.0f) * (1.0f - t);
        if (schedule == "cosine")
            return 1.0f + (baseScale - 1.0f) * 0.5f * (1.0f + MathF.Cos(MathF.PI * t));
        return baseScale;
    }

    private float[] BuildTimeSchedule(int numSteps, string schedule)
    {
        float[] tSpan = new float[numSteps + 1];
        if (schedule == "cosine")
        {
            for (int i = 0; i <= numSteps; i++)
            {
                float u = (float)i / numSteps;
                tSpan[i] = 0.5f * (1f - MathF.Cos(MathF.PI * u));
            }
        }
        else if (schedule == "logsnr")
        {
            for (int i = 0; i <= numSteps; i++)
            {
                float logSnr = 5.0f - i * (10.0f / numSteps); // Linspace from 5.0 to -5.0
                tSpan[i] = 1.0f / (1.0f + MathF.Exp(-(-logSnr / 2.0f))); // Sigmoid
            }
            tSpan[0] = 0.0f;
            tSpan[numSteps] = 1.0f;
        }
        else // uniform
        {
            for (int i = 0; i <= numSteps; i++)
            {
                tSpan[i] = (float)i / numSteps;
            }
        }
        return tSpan;
    }

    private void SolveFlowMatching(
        Span<float> speech,
        ReadOnlySpan<float> cond,
        ReadOnlySpan<float> negCond,
        int numSteps,
        float acousticCfgScale,
        float durationCfgScale,
        string cfgSchedule,
        string timeSchedule,
        int batchSize)
    {
        float[] tSpan = BuildTimeSchedule(numSteps, timeSchedule);
        float tCurr = tSpan[0];

        int totalDim = _config.AcousticDim + 2 * NumTimeBits;
        Span<float> velocity = new float[batchSize * totalDim];

        for (int i = 1; i < tSpan.Length; i++)
        {
            float dt = tSpan[i] - tCurr;
            float tVal = tCurr;
            float aCfg = ScheduledCfg(acousticCfgScale, tVal, cfgSchedule);
            float dCfg = ScheduledCfg(durationCfgScale, tVal, cfgSchedule);

            ComputeVelocity(speech, tCurr, cond, negCond, aCfg, dCfg, velocity, batchSize);

            for (int j = 0; j < speech.Length; j++)
            {
                speech[j] += dt * velocity[j];
            }

            tCurr = tSpan[i];
        }
    }

    public float[] Generate(EncoderOutput prompt, string text, InferenceOptions options)
    {
        // For testing/porting, we simulate the autoregressive generation loop.
        // In a full port, this would involve running the LLM forward pass, taking the last hidden state,
        // and using it as condition for the flow matching.

        int batchSize = 1;
        int timeDim = 2 * NumTimeBits;
        int totalDim = _config.AcousticDim + timeDim;

        // Simulate 10 steps of generation
        int numGenerateSteps = 10;
        float[] generatedSpeech = new float[numGenerateSteps * _config.AcousticDim];

        Span<float> cond = stackalloc float[batchSize * _config.HiddenSize];
        Span<float> negCond = stackalloc float[batchSize * _config.HiddenSize];
        Span<float> speech = stackalloc float[batchSize * totalDim];

        for (int step = 0; step < numGenerateSteps; step++)
        {
            // Simulate the LLM providing a condition
            cond.Fill(0.1f * step); // Dummy values
            negCond.Clear();

            // Sample initial noise
            // Random noise multiplied by NoiseTemperature
            for (int i = 0; i < speech.Length; i++)
            {
                speech[i] = 0.5f * options.NoiseTemperature; // Dummy random
            }

            // Solve ODE
            SolveFlowMatching(
                speech,
                cond,
                negCond,
                options.NumFlowMatchingSteps,
                options.AcousticCfgScale,
                options.DurationCfgScale,
                options.CfgSchedule,
                options.TimeSchedule,
                batchSize
            );

            // Extract acoustic part
            speech.Slice(0, _config.AcousticDim).CopyTo(new Span<float>(generatedSpeech, step * _config.AcousticDim, _config.AcousticDim));
        }

        return generatedSpeech;
    }
}
