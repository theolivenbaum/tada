using NAudio.Wave;
using System;
using System.IO;

namespace Tada;

public static class AudioIO
{
    public static float[] LoadWav(string filePath, out AudioFormat format)
    {
        using var reader = new WaveFileReader(filePath);
        var sampleProvider = reader.ToSampleProvider();

        format = new AudioFormat(sampleProvider.WaveFormat.SampleRate, sampleProvider.WaveFormat.Channels);

        // Use reader.Length / reader.WaveFormat.BlockAlign to get the correct number of frames, then multiply by channels.
        long sampleCount = (reader.Length / reader.WaveFormat.BlockAlign) * format.Channels;
        float[] buffer = new float[sampleCount];

        int samplesRead = sampleProvider.Read(buffer, 0, buffer.Length);

        if (samplesRead < buffer.Length)
        {
            Array.Resize(ref buffer, samplesRead);
        }

        return buffer;
    }

    public static void SaveWav(string filePath, ReadOnlySpan<float> samples, AudioFormat format)
    {
        var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(format.SampleRate, format.Channels);
        using var writer = new WaveFileWriter(filePath, waveFormat);

        writer.WriteSamples(samples.ToArray(), 0, samples.Length);
    }
}
