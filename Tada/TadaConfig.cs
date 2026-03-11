namespace Tada;

public class TadaConfig
{
    public int AcousticDim { get; set; } = 512;
    public int NumTimeClasses { get; set; } = 1024;
    public float LatentDropout { get; set; } = 0.0f;
    public float AddSemanticToCondition { get; set; } = 0.0f;
    public int ShiftAcoustic { get; set; } = 5;
    public int HiddenSize { get; set; } = 4096;
}
