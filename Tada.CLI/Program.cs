using System;
using Tada;

namespace Tada.CLI;

class Program
{
    static void Main(string[] args)
    {
        string? promptWav = null;
        string? outputWav = null;
        string? text = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--prompt-wav" && i + 1 < args.Length)
            {
                promptWav = args[++i];
            }
            else if (args[i] == "--output-wav" && i + 1 < args.Length)
            {
                outputWav = args[++i];
            }
            else if (args[i] == "--text" && i + 1 < args.Length)
            {
                text = args[++i];
            }
        }

        if (promptWav == null || outputWav == null || text == null)
        {
            Console.WriteLine("Usage: dotnet run --project Tada.CLI -- --prompt-wav <input.wav> --output-wav <output.wav> --text \"<your text>\"");
            return;
        }

        Console.WriteLine($"Loading prompt wav from: {promptWav}");
        var audioSamples = AudioIO.LoadWav(promptWav, out var format);

        Console.WriteLine("Initializing components...");
        var encoder = new Encoder(new EncoderConfig());
        var decoder = new Decoder(new DecoderConfig());
        var tadaModel = new TadaForCausalLM(new TadaConfig());
        var inferenceOptions = new InferenceOptions();

        Console.WriteLine("Encoding prompt...");
        var prompt = encoder.Encode(audioSamples, new[] { "Initial prompt text" });

        Console.WriteLine($"Generating text-to-speech for: {text}");
        var generatedAudio = tadaModel.Generate(prompt, text, inferenceOptions);

        var decodedAudio = decoder.Decode(generatedAudio);

        Console.WriteLine($"Saving output wav to: {outputWav}");
        AudioIO.SaveWav(outputWav, decodedAudio, format);

        Console.WriteLine("Done.");
    }
}
