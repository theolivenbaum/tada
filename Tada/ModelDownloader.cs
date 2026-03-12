using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Tada;

public static class ModelDownloader
{
    private static readonly HttpClient _httpClient = new HttpClient();

    public static async Task DownloadFileAsync(string repoId, string filename, string outputDirectory)
    {
        string url = $"https://huggingface.co/{repoId}/resolve/main/{filename}";
        string outputPath = Path.Combine(outputDirectory, filename);

        if (File.Exists(outputPath))
        {
            return;
        }

        string directory = Path.GetDirectoryName(outputPath)!;
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        Console.WriteLine($"Downloading {filename} from {repoId}...");

        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var contentStream = await response.Content.ReadAsStreamAsync();
        using var fileStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

        await contentStream.CopyToAsync(fileStream);
    }

    public static async Task DownloadCodecAsync(string outputDirectory = "models/tada-codec")
    {
        string repoId = "HumeAI/tada-codec";
        string[] files = new[]
        {
            "encoder/config.json",
            "encoder/model.safetensors",
            "decoder/config.json",
            "decoder/model.safetensors",
            "spkr-verf/config.json",
            "spkr-verf/model.safetensors"
        };

        foreach (var file in files)
        {
            await DownloadFileAsync(repoId, file, outputDirectory);
        }
    }

    public static async Task Download3BMLAsync(string outputDirectory = "models/tada-3b-ml")
    {
        string repoId = "HumeAI/tada-3b-ml";
        string[] files = new[]
        {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        };

        foreach (var file in files)
        {
            await DownloadFileAsync(repoId, file, outputDirectory);
        }
    }
}
