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

    /// <summary>CTC-based speech recognition (e.g. wav2vec2): a single-graph acoustic model decoded with
    /// <see cref="ModelSharp.Audio.CtcDecoder"/>. Distinct from the encoder-decoder Whisper path.</summary>
    SpeechRecognition,
    Seq2SeqGeneration,

    /// <summary>Encoder-decoder speech-to-text (Whisper): a log-mel audio encoder plus an autoregressive
    /// text decoder with a forced language/task prompt. Wired via the Whisper processor + Seq2SeqGenerator.</summary>
    SpeechToTextSeq2Seq,
}
