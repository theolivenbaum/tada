using System;
using System.Numerics.Tensors;

namespace Tada;

public class TimestepEmbedder
{
    private readonly int _hiddenSize;
    private readonly int _frequencyEmbeddingSize;

    // Weights and biases for mlp: Sequential(Linear(freq_size, hidden_size, bias=False), SiLU, Linear(hidden_size, hidden_size, bias=False))
    private readonly float[] _linear1Weight;
    private readonly float[] _linear2Weight;

    public TimestepEmbedder(int hiddenSize, int frequencyEmbeddingSize = 256)
    {
        _hiddenSize = hiddenSize;
        _frequencyEmbeddingSize = frequencyEmbeddingSize;

        _linear1Weight = new float[_hiddenSize * _frequencyEmbeddingSize];
        _linear2Weight = new float[_hiddenSize * _hiddenSize];
    }

    public void Forward(ReadOnlySpan<float> t, Span<float> output)
    {
        int batchSize = t.Length;
        if (output.Length < batchSize * _hiddenSize)
            throw new ArgumentException("Output span is too small.");

        // Step 1: Timestep Embedding
        Span<float> tFreq = stackalloc float[batchSize * _frequencyEmbeddingSize];
        TimestepEmbedding(t, tFreq, _frequencyEmbeddingSize);

        // Step 2: Linear 1
        Span<float> linear1Out = stackalloc float[batchSize * _hiddenSize];
        TensorOperations.Linear(tFreq, linear1Out, _frequencyEmbeddingSize, _hiddenSize, batchSize, _linear1Weight, ReadOnlySpan<float>.Empty);

        // Step 3: SiLU
        TensorOperations.SiLU(linear1Out, linear1Out);

        // Step 4: Linear 2
        TensorOperations.Linear(linear1Out, output, _hiddenSize, _hiddenSize, batchSize, _linear2Weight, ReadOnlySpan<float>.Empty);
    }

    private static void TimestepEmbedding(ReadOnlySpan<float> t, Span<float> embedding, int dim, int maxPeriod = 10000)
    {
        int batchSize = t.Length;
        int half = dim / 2;

        Span<float> freqs = stackalloc float[half];
        float factor = -MathF.Log(maxPeriod) / half;
        for (int i = 0; i < half; i++)
        {
            freqs[i] = MathF.Exp(factor * i);
        }

        for (int b = 0; b < batchSize; b++)
        {
            float tb = t[b];
            for (int i = 0; i < half; i++)
            {
                float arg = tb * freqs[i];
                embedding[b * dim + i] = MathF.Cos(arg);
                embedding[b * dim + half + i] = MathF.Sin(arg);
            }
            if (dim % 2 != 0)
            {
                embedding[b * dim + dim - 1] = 0f;
            }
        }
    }
}

public class FeedForwardNetwork
{
    private readonly int _embedDim;
    private readonly int _ffnDim;

    private readonly float[] _gateProjWeight;
    private readonly float[] _upProjWeight;
    private readonly float[] _downProjWeight;

    public FeedForwardNetwork(int embedDim, int ffnDim)
    {
        _embedDim = embedDim;
        _ffnDim = ffnDim;

        _gateProjWeight = new float[_ffnDim * _embedDim];
        _upProjWeight = new float[_ffnDim * _embedDim];
        _downProjWeight = new float[_embedDim * _ffnDim];
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int batchSize)
    {
        if (output.Length < batchSize * _embedDim)
            throw new ArgumentException("Output span is too small.");

        Span<float> gate = stackalloc float[batchSize * _ffnDim];
        TensorOperations.Linear(x, gate, _embedDim, _ffnDim, batchSize, _gateProjWeight, ReadOnlySpan<float>.Empty);

        Span<float> up = stackalloc float[batchSize * _ffnDim];
        TensorOperations.Linear(x, up, _embedDim, _ffnDim, batchSize, _upProjWeight, ReadOnlySpan<float>.Empty);

        TensorOperations.SiLU(gate, gate);

        // gate * up
        for (int i = 0; i < gate.Length; i++)
        {
            gate[i] *= up[i];
        }

        TensorOperations.Linear(gate, output, _ffnDim, _embedDim, batchSize, _downProjWeight, ReadOnlySpan<float>.Empty);
    }
}

public class HeadLayer
{
    private readonly int _embedDim;
    private readonly int _ffnDim;
    private readonly int _condDim;
    private readonly float _normEps;

    private readonly FeedForwardNetwork _ffn;
    private readonly float[] _normWeight;

    // adaLN_modulation = Sequential(SiLU, Linear(cond_dim, 3 * embed_dim, bias=False))
    private readonly float[] _adaLnLinearWeight;

    public HeadLayer(int embedDim, int ffnDim, int condDim, float normEps = 1e-5f)
    {
        _embedDim = embedDim;
        _ffnDim = ffnDim;
        _condDim = condDim;
        _normEps = normEps;

        _ffn = new FeedForwardNetwork(embedDim, ffnDim);
        _normWeight = new float[embedDim];
        Array.Fill(_normWeight, 1f);

        _adaLnLinearWeight = new float[3 * embedDim * condDim];
    }

    public void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> c, Span<float> output, int batchSize)
    {
        // 1. adaLN_modulation
        Span<float> adaLnOut = stackalloc float[batchSize * 3 * _embedDim];
        Span<float> siluC = stackalloc float[batchSize * _condDim];
        c.CopyTo(siluC);
        TensorOperations.SiLU(siluC, siluC);
        TensorOperations.Linear(siluC, adaLnOut, _condDim, 3 * _embedDim, batchSize, _adaLnLinearWeight, ReadOnlySpan<float>.Empty);

        // 2. Split adaLnOut into shift, scale, gate
        Span<float> shift = stackalloc float[batchSize * _embedDim];
        Span<float> scale = stackalloc float[batchSize * _embedDim];
        Span<float> gate = stackalloc float[batchSize * _embedDim];

        for (int b = 0; b < batchSize; b++)
        {
            adaLnOut.Slice(b * 3 * _embedDim, _embedDim).CopyTo(shift.Slice(b * _embedDim, _embedDim));
            adaLnOut.Slice(b * 3 * _embedDim + _embedDim, _embedDim).CopyTo(scale.Slice(b * _embedDim, _embedDim));
            adaLnOut.Slice(b * 3 * _embedDim + 2 * _embedDim, _embedDim).CopyTo(gate.Slice(b * _embedDim, _embedDim));
        }

        // 3. modulate(norm(x), shift, scale)
        Span<float> normX = stackalloc float[batchSize * _embedDim];
        for (int b = 0; b < batchSize; b++)
        {
            TensorOperations.RMSNorm(x.Slice(b * _embedDim, _embedDim), normX.Slice(b * _embedDim, _embedDim), _normWeight, _normEps);
        }

        Span<float> modulated = stackalloc float[batchSize * _embedDim];
        for (int i = 0; i < batchSize * _embedDim; i++)
        {
            modulated[i] = normX[i] * (1f + scale[i]) + shift[i];
        }

        // 4. ffn(modulated)
        Span<float> ffnOut = stackalloc float[batchSize * _embedDim];
        _ffn.Forward(modulated, ffnOut, batchSize);

        // 5. x + gate * ffn
        for (int i = 0; i < batchSize * _embedDim; i++)
        {
            output[i] = x[i] + gate[i] * ffnOut[i];
        }
    }
}

public class FinalLayer
{
    private readonly int _hiddenSize;
    private readonly int _outputSize;
    private readonly int _condSize;
    private readonly float _normEps;

    private readonly float[] _linearWeight;
    private readonly float[] _adaLnLinearWeight;

    public FinalLayer(int hiddenSize, int outputSize, int condSize, float normEps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _condSize = condSize;
        _normEps = normEps;

        _linearWeight = new float[outputSize * hiddenSize];
        _adaLnLinearWeight = new float[2 * hiddenSize * condSize];
    }

    public void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> c, Span<float> output, int batchSize)
    {
        // 1. adaLN_modulation
        Span<float> adaLnOut = stackalloc float[batchSize * 2 * _hiddenSize];
        Span<float> siluC = stackalloc float[batchSize * _condSize];
        c.CopyTo(siluC);
        TensorOperations.SiLU(siluC, siluC);
        TensorOperations.Linear(siluC, adaLnOut, _condSize, 2 * _hiddenSize, batchSize, _adaLnLinearWeight, ReadOnlySpan<float>.Empty);

        // 2. Split into shift and scale
        Span<float> shift = stackalloc float[batchSize * _hiddenSize];
        Span<float> scale = stackalloc float[batchSize * _hiddenSize];
        for (int b = 0; b < batchSize; b++)
        {
            adaLnOut.Slice(b * 2 * _hiddenSize, _hiddenSize).CopyTo(shift.Slice(b * _hiddenSize, _hiddenSize));
            adaLnOut.Slice(b * 2 * _hiddenSize + _hiddenSize, _hiddenSize).CopyTo(scale.Slice(b * _hiddenSize, _hiddenSize));
        }

        // 3. modulate(norm(x), shift, scale)
        Span<float> normX = stackalloc float[batchSize * _hiddenSize];
        for (int b = 0; b < batchSize; b++)
        {
            TensorOperations.RMSNorm(x.Slice(b * _hiddenSize, _hiddenSize), normX.Slice(b * _hiddenSize, _hiddenSize), ReadOnlySpan<float>.Empty, _normEps);
        }

        Span<float> modulated = stackalloc float[batchSize * _hiddenSize];
        for (int i = 0; i < batchSize * _hiddenSize; i++)
        {
            modulated[i] = normX[i] * (1f + scale[i]) + shift[i];
        }

        // 4. linear(modulated)
        TensorOperations.Linear(modulated, output, _hiddenSize, _outputSize, batchSize, _linearWeight, ReadOnlySpan<float>.Empty);
    }
}

public class VibeVoiceDiffusionHead
{
    private readonly int _hiddenSize;
    private readonly int _latentSize;
    private readonly int _condDim;

    private readonly float[] _noisyImagesProjWeight;
    private readonly float[] _condProjWeight;

    private readonly TimestepEmbedder _tEmbedder;
    private readonly HeadLayer[] _layers;
    private readonly FinalLayer _finalLayer;

    public VibeVoiceDiffusionHead(int hiddenSize, int latentSize, int headLayers, float headFfnRatio = 3.0f, float rmsNormEps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _latentSize = latentSize;
        _condDim = hiddenSize;

        _noisyImagesProjWeight = new float[hiddenSize * latentSize];
        _condProjWeight = new float[_condDim * hiddenSize];

        _tEmbedder = new TimestepEmbedder(_condDim);

        int ffnDim = (int)(hiddenSize * headFfnRatio);
        _layers = new HeadLayer[headLayers];
        for (int i = 0; i < headLayers; i++)
        {
            _layers[i] = new HeadLayer(hiddenSize, ffnDim, _condDim, rmsNormEps);
        }

        _finalLayer = new FinalLayer(hiddenSize, latentSize, _condDim, rmsNormEps);
    }

    public void Forward(ReadOnlySpan<float> noisyImages, ReadOnlySpan<float> timesteps, ReadOnlySpan<float> condition, Span<float> output, int batchSize)
    {
        // x = noisy_images_proj(noisy_images)
        Span<float> x = stackalloc float[batchSize * _hiddenSize];
        TensorOperations.Linear(noisyImages, x, _latentSize, _hiddenSize, batchSize, _noisyImagesProjWeight, ReadOnlySpan<float>.Empty);

        // t = t_embedder(timesteps)
        Span<float> t = stackalloc float[batchSize * _condDim];
        _tEmbedder.Forward(timesteps, t);

        // cond = cond_proj(condition)
        Span<float> cond = stackalloc float[batchSize * _condDim];
        TensorOperations.Linear(condition, cond, _hiddenSize, _condDim, batchSize, _condProjWeight, ReadOnlySpan<float>.Empty);

        // c = cond + t
        Span<float> c = stackalloc float[batchSize * _condDim];
        for (int i = 0; i < batchSize * _condDim; i++)
        {
            c[i] = cond[i] + t[i];
        }

        Span<float> xNext = stackalloc float[batchSize * _hiddenSize];
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Forward(x, c, xNext, batchSize);
            xNext.CopyTo(x);
        }

        _finalLayer.Forward(x, c, output, batchSize);
    }
}
