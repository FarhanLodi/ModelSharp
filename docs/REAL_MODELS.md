# Real-model opt-in integration tests (Phase A)

ModelSharp ships no model weights — they are large and license-encumbered, and the repo is
pure-managed with no network access. The Phase A integration tests therefore run **only when you
provide the model files yourself**. With no files present they **skip (stay green)**, exactly like
the existing `MiniLmTests`. Drop the exported files into the models directory and the same tests run
live against the real models.

## Where the files go

All real-model assets live in a single directory resolved, in order, by
`tests/ModelSharp.Tests/RealModels/RealModelAssets.cs`:

1. the `MODELSHARP_MODELS_DIR` environment variable, if set (an absolute or relative path);
2. otherwise the first `models/` directory found walking up from the test output directory
   (i.e. a repo-root `models/`);
3. otherwise `assets/models/` under the test output directory (the MiniLM layout).

Set the env var to point at a big, out-of-git drive, e.g.:

```bash
export MODELSHARP_MODELS_DIR=/data/models      # Linux/macOS
setx MODELSHARP_MODELS_DIR D:\models           # Windows
```

Each test calls `RealModelAssets.TryPath("file", out var path)`; if the file is absent the test
writes a "…skipping." line and returns. No asset, no failure.

## Exporting each model with HF Optimum

Install the exporter once:

```bash
pip install "optimum[exporters]" onnx torch transformers
```

### A1 — distilgpt2 (text generation)

Export a **non-merged** decoder-with-past (NOT `*_merged`):

```bash
optimum-cli export onnx \
  --model distilgpt2 \
  --task text-generation-with-past \
  distilgpt2-onnx/
```

Then place into the models dir:

| File in models dir          | Source                                           |
| --------------------------- | ------------------------------------------------ |
| `distilgpt2.onnx`           | `distilgpt2-onnx/decoder_model.onnx` (or `model.onnx`) |
| `distilgpt2-vocab.json`     | `distilgpt2-onnx/vocab.json`                      |
| `distilgpt2-merges.txt`     | `distilgpt2-onnx/merges.txt`                      |

Config facts the test hard-codes from `config.json`: `n_head = 12`, `n_embd = 768`
→ `head_dim = 64`, `eos/bos = 50256`. (The managed CPU engine reports no concrete past-KV shapes,
so these head dims size the empty first-pass KV tensors via `DecoderModelOptions`.)

**Asserts** (`DistilGpt2GenerationTests`):
- `DistilGpt2_Op_Coverage_Probe` — every op in the graph is in `KernelRegistry.CreateDefault()`
  (no missing ops).
- `DistilGpt2_Greedy_Decode_Is_Deterministic` — `Pipeline.Load(model, manifest).Generate("The quick
  brown fox", Greedy(16))` returns non-empty text, and two independent runs produce **identical**
  output (greedy argmax is deterministic).

### A2 — ResNet50 / MobileNet (image classification)

```bash
optimum-cli export onnx --model microsoft/resnet-50 --task image-classification resnet50-onnx/
```

| File in models dir       | Notes                                                   |
| ------------------------ | ------------------------------------------------------- |
| `resnet50.onnx`          | the exported model (224×224 NCHW ImageNet classifier)   |
| `sample.jpg`             | any image to classify                                   |
| `imagenet-labels.txt`    | *optional*, one label per line, index-aligned to output |

Preprocessing is fixed in the test: RGB, 224×224, scale to `[0,1]`, ImageNet
`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.

**Asserts** (`ImageClassifierTests`):
- op-coverage probe (no missing ops).
- `ImageClassifier_Produces_Plausible_Top1` — `Run<List<Classification>>(image)` returns a
  non-empty, descending-score list; top-1 score is in `[0,1]` and the class index is non-negative.

### A3 — YOLOv5 / YOLOv8 (object detection)

YOLOv5 (Ultralytics):

```bash
pip install ultralytics
yolo export model=yolov5su.pt format=onnx imgsz=640     # or yolov8n.pt for v8
```

| File in models dir   | Notes                                              |
| -------------------- | -------------------------------------------------- |
| `yolo.onnx`          | exported detector (640×640 NCHW)                   |
| `detect.jpg`         | any image to detect on                             |
| `coco-labels.txt`    | *optional*, one label per line                     |

Select the output layout with `MODELSHARP_YOLO_LAYOUT` (`yolov5` or `yolov8`, default `yolov5`),
which the test passes as the manifest `Extra["det_layout"]` hint:
- `yolov5` → `[1, N, 5+C]` rows `[cx, cy, w, h, obj, class…]`;
- `yolov8` → transposed `[1, 4+C, N]`, no objectness.

Preprocessing: RGB, 640×640, scale to `[0,1]`, no mean/std offset.

**Asserts** (`YoloDetectionTests`):
- op-coverage probe (no missing ops).
- `Yolo_Produces_Plausible_Boxes` — `Run<List<Detection>>(image)` returns post-NMS boxes that are
  all well-formed: `X2 ≥ X1`, `Y2 ≥ Y1`, `Score ∈ [0,1]`, `ClassId ≥ 0`.

### A4 — wav2vec2 (CTC speech recognition)

```bash
optimum-cli export onnx \
  --model facebook/wav2vec2-base-960h \
  --task automatic-speech-recognition \
  wav2vec2-onnx/
```

| File in models dir        | Notes                                                        |
| ------------------------- | ------------------------------------------------------------ |
| `wav2vec2.onnx`           | CTC model; input `input_values` `[1, samples]`, output `logits` `[1, T, V]` |
| `speech.wav`              | a short **mono, 16 kHz, 16-bit PCM** clip                    |
| `wav2vec2-vocab.json`     | the HF `{token: index}` vocab map                            |

The test feeds the **raw waveform** (zero-mean / unit-variance normalized — wav2vec2 has its own
conv feature extractor, so no mel front end is applied), reshapes `logits` `[1,T,V] → [T,V]`, and
runs `CtcDecoder.GreedyDecode` with `CtcVocabulary` (word delimiter `"|"`, blank index **0**).
To make a 16 kHz mono PCM-16 clip: `ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 speech.wav`.

**Asserts** (`Wav2Vec2CtcTests`):
- op-coverage probe (no missing ops).
- `Wav2Vec2_Transcribes_Recognizably` — greedy CTC decode yields a non-empty transcript containing
  at least one letter (a real word, not just punctuation/blanks).

## Running the live tests

With the files in place:

```bash
MODELSHARP_MODELS_DIR=/data/models dotnet test --filter "FullyQualifiedName~RealModels"
```

Without the files, the same command passes — every real-model test skips.
