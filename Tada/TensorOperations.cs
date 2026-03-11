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
}
