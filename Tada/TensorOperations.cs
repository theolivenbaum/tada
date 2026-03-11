using System;
using System.Numerics;
using System.Numerics.Tensors;

namespace Tada;

public static class TensorOperations
{
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        return TensorPrimitives.Dot(a, b);
    }

    public static void Cat(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination)
    {
        if (destination.Length < a.Length + b.Length)
            throw new ArgumentException("Destination span is too small.");

        a.CopyTo(destination.Slice(0, a.Length));
        b.CopyTo(destination.Slice(a.Length, b.Length));
    }

    public static void Cos(ReadOnlySpan<float> a, Span<float> destination)
    {
        TensorPrimitives.Cos(a, destination);
    }

    public static void Sin(ReadOnlySpan<float> a, Span<float> destination)
    {
        TensorPrimitives.Sin(a, destination);
    }

    public static void Clamp(ReadOnlySpan<float> a, float min, float max, Span<float> destination)
    {
        TensorPrimitives.Clamp(a, min, max, destination);
    }

    public static void SiLU(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");

        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            destination[i] = x * (1f / (1f + MathF.Exp(-x)));
        }
    }

    public static void GELU(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");

        const float sqrt2OverPi = 0.7978845608028654f; // MathF.Sqrt(2.0f / MathF.PI)
        const float coeff = 0.044715f;

        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            float cube = x * x * x;
            destination[i] = 0.5f * x * (1f + MathF.Tanh(sqrt2OverPi * (x + coeff * cube)));
        }
    }

    public static void LayerNorm(ReadOnlySpan<float> input, Span<float> destination, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, float eps = 1e-5f)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        if (weight.Length != input.Length || bias.Length != input.Length)
            throw new ArgumentException("Weight and bias spans must have the same length as input.");

        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }
        float mean = sum / input.Length;

        float varianceSum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            float diff = input[i] - mean;
            varianceSum += diff * diff;
        }
        float variance = varianceSum / input.Length;
        float invStdDev = 1f / MathF.Sqrt(variance + eps);

        for (int i = 0; i < input.Length; i++)
        {
            destination[i] = ((input[i] - mean) * invStdDev) * weight[i] + bias[i];
        }
    }

    public static void Softmax(ReadOnlySpan<float> input, Span<float> destination, int count)
    {
        for (int i = 0; i < input.Length; i += count)
        {
            var inSlice = input.Slice(i, count);
            var outSlice = destination.Slice(i, count);

            float max = float.NegativeInfinity;
            for (int j = 0; j < count; j++)
            {
                if (inSlice[j] > max)
                    max = inSlice[j];
            }

            float sum = 0f;
            for (int j = 0; j < count; j++)
            {
                float exp = MathF.Exp(inSlice[j] - max);
                outSlice[j] = exp;
                sum += exp;
            }

            float invSum = 1f / sum;
            for (int j = 0; j < count; j++)
            {
                outSlice[j] *= invSum;
            }
        }
    }

    public static void RMSNorm(ReadOnlySpan<float> input, Span<float> destination, ReadOnlySpan<float> weight, float eps = 1e-6f)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        if (weight.Length > 0 && weight.Length != input.Length)
            throw new ArgumentException("Weight span must have the same length as input if provided.");

        float sumSquares = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            sumSquares += input[i] * input[i];
        }
        float meanSquare = sumSquares / input.Length;
        float invRms = 1f / MathF.Sqrt(meanSquare + eps);

        if (weight.Length > 0)
        {
            for (int i = 0; i < input.Length; i++)
            {
                destination[i] = (input[i] * invRms) * weight[i];
            }
        }
        else
        {
            for (int i = 0; i < input.Length; i++)
            {
                destination[i] = input[i] * invRms;
            }
        }
    }

    public static void Conv1d(
        ReadOnlySpan<float> input,
        Span<float> output,
        int inChannels,
        int outChannels,
        int seqLen,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        int kernelSize,
        int stride,
        int padding,
        int dilation = 1)
    {
        int outSeqLen = (seqLen + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1;
        if (output.Length < outChannels * outSeqLen)
            throw new ArgumentException("Output span is too small.");

        output.Slice(0, outChannels * outSeqLen).Clear();

        for (int oc = 0; oc < outChannels; oc++)
        {
            float b = bias.Length > 0 ? bias[oc] : 0f;

            for (int t = 0; t < outSeqLen; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int inT = t * stride - padding + k * dilation;
                        if (inT >= 0 && inT < seqLen)
                        {
                            int inputIdx = ic * seqLen + inT;
                            int weightIdx = oc * inChannels * kernelSize + ic * kernelSize + k;
                            sum += input[inputIdx] * weight[weightIdx];
                        }
                    }
                }
                output[oc * outSeqLen + t] = sum;
            }
        }
    }

    public static void ConvTranspose1d(
        ReadOnlySpan<float> input,
        Span<float> output,
        int inChannels,
        int outChannels,
        int seqLen,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        int kernelSize,
        int stride,
        int padding,
        int outputPadding = 0)
    {
        int outSeqLen = (seqLen - 1) * stride - 2 * padding + kernelSize + outputPadding;
        if (output.Length < outChannels * outSeqLen)
            throw new ArgumentException("Output span is too small.");

        output.Slice(0, outChannels * outSeqLen).Clear();

        for (int oc = 0; oc < outChannels; oc++)
        {
            if (bias.Length > 0)
            {
                float b = bias[oc];
                for (int t = 0; t < outSeqLen; t++)
                {
                    output[oc * outSeqLen + t] = b;
                }
            }
        }

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int outT = t * stride - padding + k;
                        if (outT >= 0 && outT < outSeqLen)
                        {
                            int inputIdx = ic * seqLen + t;
                            int weightIdx = ic * outChannels * kernelSize + oc * kernelSize + k;
                            output[oc * outSeqLen + outT] += input[inputIdx] * weight[weightIdx];
                        }
                    }
                }
            }
        }
    }

    public static void Linear(ReadOnlySpan<float> input, Span<float> output, int inFeatures, int outFeatures, int batchSize, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias)
    {
        if (input.Length != batchSize * inFeatures)
            throw new ArgumentException("Input length does not match batchSize * inFeatures.");
        if (output.Length < batchSize * outFeatures)
            throw new ArgumentException("Output span is too small.");
        if (weight.Length != inFeatures * outFeatures)
            throw new ArgumentException("Weight length does not match inFeatures * outFeatures.");

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outFeatures; o++)
            {
                float sum = bias.Length > 0 ? bias[o] : 0f;
                int weightOffset = o * inFeatures;
                int inputOffset = b * inFeatures;

                sum += TensorPrimitives.Dot(input.Slice(inputOffset, inFeatures), weight.Slice(weightOffset, inFeatures));

                output[b * outFeatures + o] = sum;
            }
        }
    }
}
