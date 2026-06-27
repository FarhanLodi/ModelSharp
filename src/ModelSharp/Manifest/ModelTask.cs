namespace ModelSharp.Manifest;

/// <summary>The task a model performs — selects the pre/post processors.</summary>
public enum ModelTask
{
    Unknown = 0,
    ImageClassification,
    ObjectDetection,
    Segmentation,
    TextRecognition,
    Embedding,
    TextGeneration,
    SpeechRecognition,
    Seq2SeqGeneration,
}
