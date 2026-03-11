namespace Tada;

public class InferenceOptions
{
    public bool TextDoSample { get; set; } = true;
    public float TextTemperature { get; set; } = 0.6f;
    public int TextTopK { get; set; } = 0;
    public float TextTopP { get; set; } = 0.9f;
    public float TextRepetitionPenalty { get; set; } = 1.1f;
    public float AcousticCfgScale { get; set; } = 1.6f;
    public float DurationCfgScale { get; set; } = 1.0f;
    public string CfgSchedule { get; set; } = "cosine";
    public float NoiseTemperature { get; set; } = 0.9f;
    public int NumFlowMatchingSteps { get; set; } = 20;
    public string TimeSchedule { get; set; } = "logsnr";
    public int NumAcousticCandidates { get; set; } = 1;
    public string Scorer { get; set; } = "likelihood";
    public float SpkrVerificationWeight { get; set; } = 1.0f;
    public float? SpeedUpFactor { get; set; } = null;
    public string NegativeConditionSource { get; set; } = "negative_step_output";
    public float TextOnlyLogitScale { get; set; } = 0.0f;
}
