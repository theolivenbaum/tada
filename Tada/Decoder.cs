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

public class DecoderBlock
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _stride;

    private readonly Snake1d _snake1;
    private readonly float[] _convTransposeWeight;
    private readonly float[] _convTransposeBias;

    private readonly ResidualUnit _res1;
    private readonly ResidualUnit _res2;
    private readonly ResidualUnit _res3;

    public DecoderBlock(int inputDim = 16, int outputDim = 8, int stride = 1)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _stride = stride;

        _snake1 = new Snake1d(inputDim);

        int pad = (int)Math.Ceiling(stride / 2.0);
        _convTransposeWeight = new float[inputDim * outputDim * (2 * stride)];
        _convTransposeBias = new float[outputDim];

        _res1 = new ResidualUnit(outputDim, 1);
        _res2 = new ResidualUnit(outputDim, 3);
        _res3 = new ResidualUnit(outputDim, 9);
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int seqLen)
    {
        float[] s1Out = new float[_inputDim * seqLen];
        _snake1.Forward(x, s1Out, seqLen);

        int pad = (int)Math.Ceiling(_stride / 2.0);
        int ctOutSeqLen = (seqLen - 1) * _stride - 2 * pad + 2 * _stride;
        float[] ctOut = new float[_outputDim * ctOutSeqLen];
        TensorOperations.ConvTranspose1d(s1Out, ctOut, _inputDim, _outputDim, seqLen, _convTransposeWeight, _convTransposeBias, 2 * _stride, _stride, pad, 0);

        float[] r1Out = new float[_outputDim * ctOutSeqLen];
        _res1.Forward(ctOut, r1Out, ctOutSeqLen);

        float[] r2Out = new float[_outputDim * ctOutSeqLen];
        _res2.Forward(r1Out, r2Out, ctOutSeqLen);

        _res3.Forward(r2Out, output, ctOutSeqLen);
    }
}

public class DACDecoder
{
    private readonly int _inputChannel;
    private readonly int _channels;
    private readonly int[] _rates;
    private readonly int _dOut;

    private readonly float[] _firstConvWeight;
    private readonly float[] _firstConvBias;

    private readonly DecoderBlock[] _blocks;

    private readonly Snake1d _lastSnake;
    private readonly float[] _lastConvWeight;
    private readonly float[] _lastConvBias;

    public DACDecoder(int inputChannel, int channels, int[] rates, int dOut = 1)
    {
        _inputChannel = inputChannel;
        _channels = channels;
        _rates = rates;
        _dOut = dOut;

        _firstConvWeight = new float[channels * inputChannel * 7];
        _firstConvBias = new float[channels];

        _blocks = new DecoderBlock[rates.Length];
        int currentDim = channels;
        for (int i = 0; i < rates.Length; i++)
        {
            int inputDim = channels / (1 << i);
            int outputDim = channels / (1 << (i + 1));
            _blocks[i] = new DecoderBlock(inputDim, outputDim, rates[i]);
            currentDim = outputDim;
        }

        _lastSnake = new Snake1d(currentDim);
        _lastConvWeight = new float[dOut * currentDim * 7];
        _lastConvBias = new float[dOut];
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> output, int seqLen)
    {
        // 1. First conv
        int outSeqLen1 = seqLen; // padding=3, kernel=7, stride=1 -> outSeqLen = seqLen
        float[] c1Out = new float[_channels * outSeqLen1];
        TensorOperations.Conv1d(x, c1Out, _inputChannel, _channels, seqLen, _firstConvWeight, _firstConvBias, 7, 1, 3);

        // 2. Decoder blocks
        float[] currentOut = c1Out;
        int currentSeqLen = outSeqLen1;
        int currentChannels = _channels;

        for (int i = 0; i < _blocks.Length; i++)
        {
            int stride = _rates[i];
            int pad = (int)Math.Ceiling(stride / 2.0);
            int nextSeqLen = (currentSeqLen - 1) * stride - 2 * pad + 2 * stride;
            int nextChannels = currentChannels / 2;

            float[] nextOut = new float[nextChannels * nextSeqLen];
            _blocks[i].Forward(currentOut, nextOut, currentSeqLen);

            currentOut = nextOut;
            currentSeqLen = nextSeqLen;
            currentChannels = nextChannels;
        }

        // 3. Last snake + conv + tanh
        float[] snakeOut = new float[currentChannels * currentSeqLen];
        _lastSnake.Forward(currentOut, snakeOut, currentSeqLen);

        TensorOperations.Conv1d(snakeOut, output, currentChannels, _dOut, currentSeqLen, _lastConvWeight, _lastConvBias, 7, 1, 3);

        for (int i = 0; i < output.Length; i++)
        {
            output[i] = MathF.Tanh(output[i]);
        }
    }
}

public class Decoder
{
    private readonly DecoderConfig _config;

    private readonly LocalAttentionEncoder _localAttentionDecoder;
    private readonly DACDecoder _wavDecoder;
    private readonly float[] _decoderProjWeight;
    private readonly float[] _decoderProjBias;

    public Decoder(DecoderConfig config)
    {
        _config = config;

        _decoderProjWeight = new float[config.HiddenDim * config.EmbedDim];
        _decoderProjBias = new float[config.HiddenDim];

        _localAttentionDecoder = new LocalAttentionEncoder(
            config.HiddenDim,
            config.NumAttnLayers,
            config.NumAttnHeads,
            config.AttnDimFeedforward
        );

        int[] strides = new[] { 4, 4, 5, 6 }; // Default from reference config
        _wavDecoder = new DACDecoder(
            inputChannel: config.HiddenDim,
            channels: 1536, // wav_decoder_channels
            rates: strides
        );
    }

    public float[] Decode(ReadOnlySpan<float> input)
    {
        // For testing/porting purposes, we assume input is [seqLen, embedDim] for a batch of 1.
        int seqLen = input.Length / _config.EmbedDim;

        // 1. Decoder projection
        Span<float> decoderInput = new float[seqLen * _config.HiddenDim];
        TensorOperations.Linear(input, decoderInput, _config.EmbedDim, _config.HiddenDim, seqLen, _decoderProjWeight, _decoderProjBias);

        // 2. LocalAttentionEncoder (acting as decoder attention)
        Span<float> decodedExpanded = new float[seqLen * _config.HiddenDim];
        _localAttentionDecoder.Forward(decoderInput, decodedExpanded, 1, seqLen); // Batch size = 1

        // 3. Transpose decodedExpanded from [seqLen, channels] to [channels, seqLen] for DACDecoder
        Span<float> decodedTransposed = new float[_config.HiddenDim * seqLen];
        for (int c = 0; c < _config.HiddenDim; c++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                decodedTransposed[c * seqLen + t] = decodedExpanded[t * _config.HiddenDim + c];
            }
        }

        // 4. DACDecoder
        // Calculate output length for DACDecoder
        int currentSeqLen = seqLen;
        int[] strides = new[] { 4, 4, 5, 6 };
        for (int i = 0; i < strides.Length; i++)
        {
            int stride = strides[i];
            int pad = (int)Math.Ceiling(stride / 2.0);
            currentSeqLen = (currentSeqLen - 1) * stride - 2 * pad + 2 * stride;
        }

        float[] xRec = new float[currentSeqLen]; // dOut is 1
        _wavDecoder.Forward(decodedTransposed, xRec, seqLen);

        return xRec;
    }
}
