using System;
using System.Numerics.Tensors;

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

public class Snake1d
{
    private readonly int _channels;
    private readonly float[] _alpha;

    public Snake1d(int channels)
    {
        _channels = channels;
        _alpha = new float[channels];
        Array.Fill(_alpha, 1f); // Default initialization
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int seqLen)
    {
        // output = x + (1/alpha) * sin^2(alpha * x)
        for (int c = 0; c < _channels; c++)
        {
            float a = _alpha[c];
            float invA = 1f / a;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = c * seqLen + t;
                float x = input[idx];
                float sin = MathF.Sin(a * x);
                output[idx] = x + invA * sin * sin;
            }
        }
    }
}

public class ResidualUnit
{
    private readonly int _dim;
    private readonly int _dilation;

    private readonly Snake1d _snake1;
    private readonly float[] _conv1Weight;
    private readonly float[] _conv1Bias;

    private readonly Snake1d _snake2;
    private readonly float[] _conv2Weight;
    private readonly float[] _conv2Bias;

    public ResidualUnit(int dim, int dilation = 1)
    {
        _dim = dim;
        _dilation = dilation;
        int pad = (6 * dilation) / 2;

        _snake1 = new Snake1d(dim);
        _conv1Weight = new float[dim * dim * 7];
        _conv1Bias = new float[dim];

        _snake2 = new Snake1d(dim);
        _conv2Weight = new float[dim * dim * 1];
        _conv2Bias = new float[dim];
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int seqLen)
    {
        int pad = (6 * _dilation) / 2;

        float[] s1Out = new float[_dim * seqLen];
        _snake1.Forward(x, s1Out, seqLen);

        float[] c1Out = new float[_dim * seqLen]; // Padding applied internally in Conv1d
        TensorOperations.Conv1d(s1Out, c1Out, _dim, _dim, seqLen, _conv1Weight, _conv1Bias, 7, 1, pad, _dilation);

        float[] s2Out = new float[_dim * seqLen];
        _snake2.Forward(c1Out, s2Out, seqLen);

        float[] c2Out = new float[_dim * seqLen];
        TensorOperations.Conv1d(s2Out, c2Out, _dim, _dim, seqLen, _conv2Weight, _conv2Bias, 1, 1, 0, 1);

        for (int i = 0; i < x.Length; i++)
        {
            output[i] = x[i] + c2Out[i];
        }
    }
}

public class EncoderBlock
{
    private readonly int _dim;
    private readonly int _stride;

    private readonly ResidualUnit _res1;
    private readonly ResidualUnit _res2;
    private readonly ResidualUnit _res3;

    private readonly Snake1d _snake;
    private readonly float[] _convWeight;
    private readonly float[] _convBias;

    public EncoderBlock(int dim = 16, int stride = 1)
    {
        _dim = dim;
        _stride = stride;
        int halfDim = dim / 2;

        _res1 = new ResidualUnit(halfDim, 1);
        _res2 = new ResidualUnit(halfDim, 3);
        _res3 = new ResidualUnit(halfDim, 9);

        _snake = new Snake1d(halfDim);

        int pad = (int)Math.Ceiling(stride / 2.0);
        _convWeight = new float[dim * halfDim * (2 * stride)];
        _convBias = new float[dim];
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int seqLen)
    {
        int halfDim = _dim / 2;
        int outSeqLen = (seqLen + 2 * (int)Math.Ceiling(_stride / 2.0) - (2 * _stride - 1) - 1) / _stride + 1;

        // Use arrays instead of stackalloc for potentially large sequences to avoid stack overflow
        float[] r1Out = new float[halfDim * seqLen];
        _res1.Forward(x, r1Out, seqLen);

        float[] r2Out = new float[halfDim * seqLen];
        _res2.Forward(r1Out, r2Out, seqLen);

        float[] r3Out = new float[halfDim * seqLen];
        _res3.Forward(r2Out, r3Out, seqLen);

        float[] snakeOut = new float[halfDim * seqLen];
        _snake.Forward(r3Out, snakeOut, seqLen);

        int pad = (int)Math.Ceiling(_stride / 2.0);
        TensorOperations.Conv1d(snakeOut, output, halfDim, _dim, seqLen, _convWeight, _convBias, 2 * _stride, _stride, pad, 1);
    }
}

public class WavEncoder
{
    private readonly int[] _strides;
    private readonly int _dLatent;
    private readonly int _initialDModel = 64;

    private readonly float[] _firstConvWeight;
    private readonly float[] _firstConvBias;

    private readonly EncoderBlock[] _blocks;

    private readonly Snake1d _lastSnake;
    private readonly float[] _lastConvWeight;
    private readonly float[] _lastConvBias;

    public int EncDim { get; }

    public WavEncoder(int dModel = 64, int[]? strides = null, int dLatent = 64)
    {
        _initialDModel = dModel;
        _strides = strides ?? new[] { 2, 4, 8, 8 };
        _dLatent = dLatent;

        _firstConvWeight = new float[dModel * 1 * 7];
        _firstConvBias = new float[dModel];

        _blocks = new EncoderBlock[_strides.Length];
        int currentDim = dModel;
        for (int i = 0; i < _strides.Length; i++)
        {
            currentDim *= 2;
            _blocks[i] = new EncoderBlock(currentDim, _strides[i]);
        }
        EncDim = currentDim;

        _lastSnake = new Snake1d(currentDim);
        _lastConvWeight = new float[dLatent * currentDim * 3];
        _lastConvBias = new float[dLatent];
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int seqLen)
    {
        // 1. First conv
        int outSeqLen1 = seqLen; // padding=3, kernel=7, stride=1 -> outSeqLen = seqLen
        float[] c1Out = new float[_initialDModel * outSeqLen1];
        TensorOperations.Conv1d(x, c1Out, 1, _initialDModel, seqLen, _firstConvWeight, _firstConvBias, 7, 1, 3);

        // 2. Encoder blocks
        float[] currentOut = c1Out;
        int currentSeqLen = outSeqLen1;
        int currentChannels = _initialDModel;

        for (int i = 0; i < _blocks.Length; i++)
        {
            int stride = _strides[i];
            int pad = (int)Math.Ceiling(stride / 2.0);
            int nextSeqLen = (currentSeqLen + 2 * pad - (2 * stride - 1) - 1) / stride + 1;
            int nextChannels = currentChannels * 2;

            float[] nextOut = new float[nextChannels * nextSeqLen];
            _blocks[i].Forward(currentOut, nextOut, currentSeqLen);

            currentOut = nextOut;
            currentSeqLen = nextSeqLen;
            currentChannels = nextChannels;
        }

        // 3. Last snake + conv
        float[] snakeOut = new float[currentChannels * currentSeqLen];
        _lastSnake.Forward(currentOut, snakeOut, currentSeqLen);

        // padding=1, kernel=3, stride=1 -> outSeqLen = currentSeqLen
        TensorOperations.Conv1d(snakeOut, output, currentChannels, _dLatent, currentSeqLen, _lastConvWeight, _lastConvBias, 3, 1, 1);
    }
}

public class LocalSelfAttention
{
    private readonly int _dModel;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _invSqrtHeadDim;

    private readonly float[] _qkvWeight;
    private readonly float[] _qkvBias;
    private readonly float[] _outProjWeight;
    private readonly float[] _outProjBias;

    private readonly float[] _layerNormWeight;
    private readonly float[] _layerNormBias;

    private readonly float[] _ropeFreqsCos;
    private readonly float[] _ropeFreqsSin;

    public LocalSelfAttention(int dModel, int numHeads = 8, int maxSeqLen = 8192)
    {
        if (dModel % numHeads != 0)
            throw new ArgumentException("d_model must be divisible by num_heads");

        _dModel = dModel;
        _numHeads = numHeads;
        _headDim = dModel / numHeads;
        _invSqrtHeadDim = 1f / MathF.Sqrt(_headDim);

        _qkvWeight = new float[3 * dModel * dModel];
        _qkvBias = new float[3 * dModel];
        _outProjWeight = new float[dModel * dModel];
        _outProjBias = new float[dModel];

        _layerNormWeight = new float[dModel];
        _layerNormBias = new float[dModel];
        Array.Fill(_layerNormWeight, 1f);

        _ropeFreqsCos = new float[maxSeqLen * _headDim / 2];
        _ropeFreqsSin = new float[maxSeqLen * _headDim / 2];

        ComputeRopeFreqs(_headDim, maxSeqLen, _ropeFreqsCos, _ropeFreqsSin);
    }

    private static void ComputeRopeFreqs(int headDim, int maxSeqLen, float[] cosArr, float[] sinArr)
    {
        int halfHeadDim = headDim / 2;
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < halfHeadDim; i++)
            {
                float invFreq = 1f / MathF.Pow(10000.0f, (float)(2 * i) / headDim);
                float arg = pos * invFreq;
                cosArr[pos * halfHeadDim + i] = MathF.Cos(arg);
                sinArr[pos * halfHeadDim + i] = MathF.Sin(arg);
            }
        }
    }

    private void ApplyRope(Span<float> x, int batchSize, int seqLen)
    {
        int halfHeadDim = _headDim / 2;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int i = 0; i < halfHeadDim; i++)
                    {
                        int idx0 = b * _numHeads * seqLen * _headDim + h * seqLen * _headDim + s * _headDim + i * 2;
                        int idx1 = idx0 + 1;

                        float x0 = x[idx0];
                        float x1 = x[idx1];

                        float cos = _ropeFreqsCos[s * halfHeadDim + i];
                        float sin = _ropeFreqsSin[s * halfHeadDim + i];

                        x[idx0] = x0 * cos - x1 * sin;
                        x[idx1] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int batchSize, int seqLen, ReadOnlySpan<bool> mask = default)
    {
        // Compute QKV
        float[] qkv = new float[batchSize * seqLen * 3 * _dModel];
        TensorOperations.Linear(x, qkv, _dModel, 3 * _dModel, batchSize * seqLen, _qkvWeight, _qkvBias);

        // Separate Q, K, V and apply RoPE to Q and K
        float[] q = new float[batchSize * _numHeads * seqLen * _headDim];
        float[] k = new float[batchSize * _numHeads * seqLen * _headDim];
        float[] v = new float[batchSize * _numHeads * seqLen * _headDim];

        Span<float> qkvSpan = qkv.AsSpan();
        Span<float> qSpan = q.AsSpan();
        Span<float> kSpan = k.AsSpan();
        Span<float> vSpan = v.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int qkvOffset = b * seqLen * 3 * _dModel + s * 3 * _dModel + h * _headDim;
                    int targetOffset = b * _numHeads * seqLen * _headDim + h * seqLen * _headDim + s * _headDim;

                    qkvSpan.Slice(qkvOffset, _headDim).CopyTo(qSpan.Slice(targetOffset, _headDim));
                    qkvSpan.Slice(qkvOffset + _dModel, _headDim).CopyTo(kSpan.Slice(targetOffset, _headDim));
                    qkvSpan.Slice(qkvOffset + 2 * _dModel, _headDim).CopyTo(vSpan.Slice(targetOffset, _headDim));
                }
            }
        }

        ApplyRope(qSpan, batchSize, seqLen);
        ApplyRope(kSpan, batchSize, seqLen);

        // Attention scores
        float[] attnScores = new float[batchSize * _numHeads * seqLen * seqLen];
        Span<float> attnScoresSpan = attnScores.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        int qOffset = b * _numHeads * seqLen * _headDim + h * seqLen * _headDim + i * _headDim;
                        int kOffset = b * _numHeads * seqLen * _headDim + h * seqLen * _headDim + j * _headDim;

                        float dot = TensorPrimitives.Dot(qSpan.Slice(qOffset, _headDim), kSpan.Slice(kOffset, _headDim));

                        bool isMasked = false;
                        if (mask.Length > 0)
                        {
                            // Mask is (batch, seq, seq) or (seq, seq)
                            if (mask.Length == seqLen * seqLen)
                                isMasked = mask[i * seqLen + j];
                            else
                                isMasked = mask[b * seqLen * seqLen + i * seqLen + j];
                        }

                        attnScoresSpan[b * _numHeads * seqLen * seqLen + h * seqLen * seqLen + i * seqLen + j] = isMasked ? float.NegativeInfinity : dot * _invSqrtHeadDim;
                    }
                }
            }
        }

        // Softmax
        TensorOperations.Softmax(attnScoresSpan, attnScoresSpan, seqLen);

        // Attn output
        float[] attnOut = new float[batchSize * seqLen * _dModel];
        Span<float> attnOutSpan = attnOut.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    int outOffset = b * seqLen * _dModel + i * _dModel + h * _headDim;

                    for (int j = 0; j < seqLen; j++)
                    {
                        float score = attnScoresSpan[b * _numHeads * seqLen * seqLen + h * seqLen * seqLen + i * seqLen + j];
                        int vOffset = b * _numHeads * seqLen * _headDim + h * seqLen * _headDim + j * _headDim;

                        for (int d = 0; d < _headDim; d++)
                        {
                            attnOutSpan[outOffset + d] += score * vSpan[vOffset + d];
                        }
                    }
                }
            }
        }

        // Out proj
        float[] projOut = new float[batchSize * seqLen * _dModel];
        Span<float> projOutSpan = projOut.AsSpan();
        TensorOperations.Linear(attnOutSpan, projOutSpan, _dModel, _dModel, batchSize * seqLen, _outProjWeight, _outProjBias);

        // Residual + LayerNorm
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = b * seqLen * _dModel + s * _dModel;
                for (int d = 0; d < _dModel; d++)
                {
                    projOutSpan[offset + d] += x[offset + d];
                }
                TensorOperations.LayerNorm(projOutSpan.Slice(offset, _dModel), output.Slice(offset, _dModel), _layerNormWeight, _layerNormBias);
            }
        }
    }
}

public class LocalAttentionEncoderLayer
{
    private readonly int _dModel;
    private readonly LocalSelfAttention _selfAttn;

    // FFN
    private readonly float[] _ffnLinear1Weight;
    private readonly float[] _ffnLinear1Bias;
    private readonly float[] _ffnLinear2Weight;
    private readonly float[] _ffnLinear2Bias;

    private readonly float[] _layerNormWeight;
    private readonly float[] _layerNormBias;

    public LocalAttentionEncoderLayer(int dModel, int numHeads = 8, int dFf = -1, int maxSeqLen = 8192)
    {
        _dModel = dModel;
        if (dFf == -1) dFf = 4 * dModel;

        _selfAttn = new LocalSelfAttention(dModel, numHeads, maxSeqLen);

        _ffnLinear1Weight = new float[dFf * dModel];
        _ffnLinear1Bias = new float[dFf];
        _ffnLinear2Weight = new float[dModel * dFf];
        _ffnLinear2Bias = new float[dModel];

        _layerNormWeight = new float[dModel];
        _layerNormBias = new float[dModel];
        Array.Fill(_layerNormWeight, 1f);
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int batchSize, int seqLen, ReadOnlySpan<bool> mask = default)
    {
        float[] attnOut = new float[batchSize * seqLen * _dModel];
        Span<float> attnOutSpan = attnOut.AsSpan();
        _selfAttn.Forward(x, attnOutSpan, batchSize, seqLen, mask);

        int dFf = _ffnLinear1Bias.Length;
        float[] ffn1 = new float[batchSize * seqLen * dFf];
        Span<float> ffn1Span = ffn1.AsSpan();
        TensorOperations.Linear(attnOutSpan, ffn1Span, _dModel, dFf, batchSize * seqLen, _ffnLinear1Weight, _ffnLinear1Bias);

        TensorOperations.GELU(ffn1Span, ffn1Span);

        float[] ffn2 = new float[batchSize * seqLen * _dModel];
        Span<float> ffn2Span = ffn2.AsSpan();
        TensorOperations.Linear(ffn1Span, ffn2Span, dFf, _dModel, batchSize * seqLen, _ffnLinear2Weight, _ffnLinear2Bias);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = b * seqLen * _dModel + s * _dModel;
                for (int d = 0; d < _dModel; d++)
                {
                    ffn2Span[offset + d] += attnOutSpan[offset + d];
                }
                TensorOperations.LayerNorm(ffn2Span.Slice(offset, _dModel), output.Slice(offset, _dModel), _layerNormWeight, _layerNormBias);
            }
        }
    }
}

public class LocalAttentionEncoder
{
    private readonly int _dModel;
    private readonly LocalAttentionEncoderLayer[] _layers;

    private readonly float[] _finalNormWeight;
    private readonly float[] _finalNormBias;

    public LocalAttentionEncoder(int dModel, int numLayers = 4, int numHeads = 8, int dFf = -1)
    {
        _dModel = dModel;

        _layers = new LocalAttentionEncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _layers[i] = new LocalAttentionEncoderLayer(dModel, numHeads, dFf);
        }

        _finalNormWeight = new float[dModel];
        _finalNormBias = new float[dModel];
        Array.Fill(_finalNormWeight, 1f);
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int batchSize, int seqLen, ReadOnlySpan<bool> mask = default)
    {
        Span<float> current = new float[batchSize * seqLen * _dModel];
        x.CopyTo(current);

        Span<float> next = new float[batchSize * seqLen * _dModel];

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Forward(current, next, batchSize, seqLen, mask);
            next.CopyTo(current);
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = b * seqLen * _dModel + s * _dModel;
                TensorOperations.LayerNorm(current.Slice(offset, _dModel), output.Slice(offset, _dModel), _finalNormWeight, _finalNormBias);
            }
        }
    }
}

public class Encoder
{
    private readonly EncoderConfig _config;

    private readonly WavEncoder _wavEncoder;
    private readonly LocalAttentionEncoder _localAttentionEncoder;

    private readonly float[] _hiddenLinearWeight;
    private readonly float[] _hiddenLinearBias;

    public Encoder(EncoderConfig config)
    {
        _config = config;

        _wavEncoder = new WavEncoder(64, config.Strides, config.HiddenDim);

        _localAttentionEncoder = new LocalAttentionEncoder(
            config.HiddenDim,
            config.NumAttnLayers,
            config.NumAttnHeads,
            4096 // _config.AttnDimFeedforward is missing, defaulting
        );

        if (config.HiddenDim != config.EmbedDim)
        {
            _hiddenLinearWeight = new float[config.EmbedDim * config.HiddenDim];
            _hiddenLinearBias = new float[config.EmbedDim];
        }
        else
        {
            _hiddenLinearWeight = Array.Empty<float>();
            _hiddenLinearBias = Array.Empty<float>();
        }
    }

    public EncoderOutput Encode(ReadOnlySpan<float> audioSamples, string[] text)
    {
        // 1. WavEncoder forward pass
        // Calculate lengths
        int seqLen = audioSamples.Length;
        int outSeqLen1 = seqLen;
        int currentSeqLen = outSeqLen1;

        for (int i = 0; i < _config.Strides.Length; i++)
        {
            int stride = _config.Strides[i];
            int pad = (int)Math.Ceiling(stride / 2.0);
            currentSeqLen = (currentSeqLen + 2 * pad - (2 * stride - 1) - 1) / stride + 1;
        }

        Span<float> encOut = new float[_config.HiddenDim * currentSeqLen];
        _wavEncoder.Forward(audioSamples, encOut, seqLen);

        // Transpose encOut from [channels, seqLen] to [seqLen, channels]
        Span<float> encOutTransposed = new float[currentSeqLen * _config.HiddenDim];
        for (int c = 0; c < _config.HiddenDim; c++)
        {
            for (int t = 0; t < currentSeqLen; t++)
            {
                encOutTransposed[t * _config.HiddenDim + c] = encOut[c * currentSeqLen + t];
            }
        }

        // 2. LocalAttentionEncoder forward pass
        Span<float> attnOut = new float[currentSeqLen * _config.HiddenDim];
        _localAttentionEncoder.Forward(encOutTransposed, attnOut, 1, currentSeqLen); // Batch size = 1

        // 3. Hidden linear projection (if needed)
        Span<float> finalOut;
        if (_hiddenLinearWeight.Length > 0)
        {
            finalOut = new float[currentSeqLen * _config.EmbedDim];
            TensorOperations.Linear(attnOut, finalOut, _config.HiddenDim, _config.EmbedDim, currentSeqLen, _hiddenLinearWeight, _hiddenLinearBias);
        }
        else
        {
            finalOut = attnOut;
        }

        return new EncoderOutput
        {
            Audio = audioSamples.ToArray(),
            AudioLen = new float[] { audioSamples.Length },
            Text = text,
            // Keeping token logic stubbed as requested for initial porting (audio only focused)
        };
    }
}
