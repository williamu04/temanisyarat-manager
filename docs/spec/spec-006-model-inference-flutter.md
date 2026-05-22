# TFLite Model Inference in Flutter (Real-Time BISINDO Recognition)

## Overview

Integrate the self-contained `model_raw_fp16.tflite` into the TemanIsyarat
Flutter app for real-time BISINDO word recognition. The app captures
MediaPipe hand + pose landmarks from the camera, accumulates a 125-frame
sliding window, runs TFLite inference every frame, and displays the
predicted word with confidence.

### Self-contained model

The model (`output/model_raw_fp16.tflite`, 2.55 MB) accepts **raw landmarks**
directly — no external preprocessing needed:

| Property | Value |
|----------|-------|
| Input shape | `(1, 125, 153)` float32 |
| Output shape | `(1, 20)` float32 (logits) |
| Quantization | FP16 |
| Runtime deps | Flex delegate (`tensorflow-lite-select-tf-ops`) |

It bakes in:
- NaN detection (`x != x`  →  mask = 0 for missing landmarks)
- Zeroing of NaN positions
- Normalization with dataset mean/std (baked as constants)
- Feature/mask concatenation

## Architecture

```
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│  Camera      │───▶│  MediaPipe Pose +    │───▶│  LandmarkBuffer (125)   │
│  (30 fps)    │    │  Hand Landmarker     │    │  ring: Float32List      │
└──────────────┘    └──────────────────────┘    │  125 × 153 = 19125      │
                                                └──────────┬──────────────┘
                                                           │ every frame
                                                           ▼
                                                ┌─────────────────────────┐
                                                │  TFLite Interpreter     │
                                                │  (model_raw_fp16.tflite)│
                                                └──────────┬──────────────┘
                                                           ▼
                                                ┌─────────────────────────┐
                                                │  Post-processing        │
                                                │  - softmax              │
                                                │  - confidence filter    │
                                                │  - temporal smoothing   │
                                                └──────────┬──────────────┘
                                                           ▼
                                                ┌─────────────────────────┐
                                                │  UI: predicted word     │
                                                │  + confidence bar       │
                                                └─────────────────────────┘
```

## Data Flow (per frame)

### 1. MediaPipe Landmark Extraction

The model expects **153 floats per frame**:

```
pose_landmarks (9 landmarks × 3 coords)  = 27
  + hand_landmarks_left (21 × 3)         = 63
  + hand_landmarks_right (21 × 3)        = 63
  ─────────────────────────────────────
  total                                  = 153
```

**Pose landmark indices** (MediaPipe Pose 33-keypoint model):

```dart
const poseIndices = [0, 11, 12, 13, 14, 15, 16, 23, 24];
// 0 = nose
// 11 = left_shoulder,   12 = right_shoulder
// 13 = left_elbow,      14 = right_elbow
// 15 = left_wrist,      16 = right_wrist
// 23 = left_hip,        24 = right_hip
```

**Hand skeleton connections** (for overlay visualization — not used by the
model, which only consumes the 21 raw landmarks per hand):

```dart
const handConnections = [
  (0, 1), (1, 2), (2, 3), (3, 4),                            // thumb
  (0, 5), (5, 6), (6, 7), (7, 8),                            // index
  (0, 9), (9, 10), (10, 11), (11, 12),                       // middle
  (0, 13), (13, 14), (14, 15), (15, 16),                     // ring
  (0, 17), (17, 18), (18, 19), (19, 20),                     // pinky
  (5, 9), (9, 13), (13, 17),                                  // palm
];
```

**NaN handling:** MediaPipe outputs `(x, y, z)` normalized to `[0, 1]`.
If a hand is not detected, the corresponding 21 landmarks should be set
to `NaN` (the model's internal NaN detection will mask them out).

### 2. Ring Buffer

```dart
class LandmarkBuffer {
  static const int maxLength = 125;
  static const int frameDim = 153; // 27 pose + 63 left + 63 right
  static const int flatLength = maxLength * frameDim; // 19125

  final Float32List _buffer = Float32List(flatLength);
  int _count = 0;

  /// Add one frame of 153 landmarks. Replaces oldest when full.
  void addFrame(Float32List frame) {
    assert(frame.length == frameDim);
    final offset = _count % maxLength;
    _buffer.setRange(offset * frameDim, (offset + 1) * frameDim, frame);
    if (_count < maxLength) _count++;
  }

  /// Returns the full 125-frame sliding window as a contiguous Float32List.
  /// Frames are ordered chronologically (oldest first).
  Float32List getWindow() {
    if (_count < maxLength) {
      // Not full yet — fill remaining slots with NaN
      final out = Float32List(flatLength);
      out.fillRange(0, flatLength, double.nan);
      out.setRange(0, _count * frameDim, _buffer, 0);
      return out;
    }
    // Rotate so oldest frame is at index 0
    final start = _count % maxLength;
    final out = Float32List(flatLength);
    out.setRange(0, (maxLength - start) * frameDim, _buffer, start * frameDim);
    out.setRange((maxLength - start) * frameDim, flatLength, _buffer, 0);
    return out;
  }

  bool get isReady => _count >= maxLength;
  int get frameCount => _count < maxLength ? _count : maxLength;
  void reset() { _count = 0; }
}
```

### 3. TFLite Inference

```dart
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

class SignLanguageInterpreter {
  late final Interpreter _interpreter;

  Future<void> load() async {
    // Use XNNPack delegate for CPU acceleration
    final interpreterOptions = InterpreterOptions()
      ..useXNNPack = true;

    _interpreter = await Interpreter.fromAsset(
      'models/model_raw_fp16.tflite',
      options: interpreterOptions,
    );

    // Verify input/output shapes
    assert(_interpreter.getInputTensors().first.shape
        .listEquals([1, 125, 153]));
    assert(_interpreter.getOutputTensors().first.shape
        .listEquals([1, 20]));
  }

  /// Run inference. [input] is a flat Float32List of 19125 values.
  /// Returns 20 logits as Float32List.
  Float32List predict(Float32List input) {
    assert(input.length == 19125);

    final output = Float32List(20);
    // Reshape input to [1, 125, 153] for the interpreter
    final inputTensor = input.reshape([1, 125, 153]);
    final outputTensor = [output.reshape([1, 20])];

    _interpreter.runForMultipleInputs([inputTensor], outputTensor);
    return output;
  }

  void dispose() {
    _interpreter.close();
  }
}
```

**Flex delegate note:** The GRU layers require Flex ops
(`FlexTensorListReserve`, `FlexTensorListSetItem`, `FlexTensorListStack`).
On Android, add to `android/app/build.gradle`:

```kotlin
dependencies {
    // TFLite with Flex (Select TF ops)
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")
}
```

On iOS, `tflite_flutter` doesn't support iOS. Use Swift TensorFlow Lite
with the C API and link the Flex delegate manually, or use
`tensorflow_lite_flutter` which supports iOS.

### 4. Post-Processing

```dart
class PredictionProcessor {
  final List<String> classes;
  final List<int> _history = [];
  static const int historySize = 10;

  PredictionProcessor(this.classes);

  /// Return predicted label (or null if below threshold).
  String? process(Float32List logits, {double confidenceThreshold = 0.7}) {
    // Softmax
    final max = logits.reduce((a, b) => a > b ? a : b);
    final exps = logits.map((x) => (x - max).exp()).toList();
    final sum = exps.reduce((a, b) => a + b);
    final probs = exps.map((e) => e / sum).toList();

    // Confidence filter
    final bestProb = probs.reduce((a, b) => a > b ? a : b);
    if (bestProb < confidenceThreshold) return null;

    final predIdx = probs.indexOf(bestProb);
    _history.add(predIdx);
    if (_history.length > historySize) _history.removeAt(0);

    // Temporal majority vote (60% agreement required)
    final counts = <int, int>{};
    for (final p in _history) {
      counts[p] = (counts[p] ?? 0) + 1;
    }
    final bestEntry = counts.entries.reduce((a, b) => a.value > b.value ? a : b);
    if (bestEntry.value < historySize * 0.6) return null;

    return classes[bestEntry.key];
  }

  void reset() => _history.clear();
}
```

### 5. Main Loop

```dart
class SignLanguageRecognitionLoop {
  final LandmarkBuffer _buffer = LandmarkBuffer();
  late final SignLanguageInterpreter _interpreter;
  late final PredictionProcessor _processor;

  Future<void> init() async {
    _interpreter = SignLanguageInterpreter();
    await _interpreter.load();
    _processor = PredictionProcessor(_classList); // 20 labels
  }

  /// Called from MediaPipe callback every frame.
  /// [landmarks] is a flat Float32List of 153 values.
  String? onFrame(Float32List landmarks) {
    _buffer.addFrame(landmarks);
    if (!_buffer.isReady) return null; // warm-up: not enough frames

    final window = _buffer.getWindow();
    final logits = _interpreter.predict(window);
    return _processor.process(logits);
  }

  void dispose() {
    _interpreter.dispose();
  }
}
```

## Landmark Assembly (from MediaPipe outputs)

### Dart helper to build the 153-float frame vector

```dart
Float32List assembleFrame({
  required List<MediaPipePose> pose,       // 33 landmarks (or subset)
  required List<MediaPipeHand>? leftHand,  // 21 or null
  required List<MediaPipeHand>? rightHand, // 21 or null
  List<int> poseIndices = const [0, 11, 12, 13, 14, 15, 16, 23, 24],
}) {
  final frame = Float32List(153);
  int idx = 0;

  // 9 pose landmarks × 3 = 27 floats
  for (final pi in poseIndices) {
    final lm = pose[pi];
    frame[idx++] = lm.x;
    frame[idx++] = lm.y;
    frame[idx++] = lm.z;
  }

  // Left hand: 21 × 3 = 63 floats (NaN if not detected)
  if (leftHand != null && leftHand.length == 21) {
    for (final lm in leftHand) {
      frame[idx++] = lm.x;
      frame[idx++] = lm.y;
      frame[idx++] = lm.z;
    }
  } else {
    idx += 63;
    // Already 0.0 from Float32List; set to NaN for proper masking:
    for (int i = idx - 63; i < idx; i++) frame[i] = double.nan;
  }

  // Right hand: 21 × 3 = 63 floats
  if (rightHand != null && rightHand.length == 21) {
    for (final lm in rightHand) {
      frame[idx++] = lm.x;
      frame[idx++] = lm.y;
      frame[idx++] = lm.z;
    }
  } else {
    for (int i = idx; i < idx + 63; i++) frame[i] = double.nan;
    idx += 63;
  }

  return frame;
}
```

> **Important:** The hand order (left vs right) must match the training
> data. The .npz files store hands as `(T, 2, 21, 3)` where axis 1 is
> `[left, right]`. If your app swaps them, predictions will be wrong.

## Class Labels (20 BISINDO words)

```dart
const classLabels = [
  "aku", "apel", "ayah", "besok", "buku",
  "dia", "dua", "hari ini", "ibu", "kamu",
  "kuning", "maaf", "merah", "nama", "pisang",
  "salam", "satu", "teman", "terima kasih", "tiga",
];
```

## UI Flow

```
┌───────────────────────────────────────────────┐
│  Camera Preview (Full screen)                 │
│                                               │
│           ┌───────────────┐                   │
│           │  TERIMA KASIH │  ← predicted word │
│           │  ████████░░ 92%│  ← confidence    │
│           └───────────────┘                   │
│                                               │
│  [⚙ Settings]  [🔄 History]                   │
└───────────────────────────────────────────────┘
```

States:

| Phase | Buffer state | UI |
|-------|-------------|-----|
| Warm-up | `< 125 frames` | Spinner + "Memulai..." |
| Ready | `125 frames` | Show prediction every frame |
| Silence | confidence < threshold | Show nothing (or faint placeholder) |
| Cooldown | after word displayed | Hold result for 1.5s before clearing |

## Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `lib/services/landmark_buffer.dart` | Ring buffer (125 × 153) |
| 2 | `lib/services/tflite_interpreter.dart` | TFLite model loading + inference |
| 3 | `lib/services/prediction_processor.dart` | Softmax + temporal smoothing |
| 4 | `lib/services/sign_language_engine.dart` | Orchestrator (buffer + inference + post-process) |
| 5 | `lib/utils/landmark_assembler.dart` | Build 153-float frame from MediaPipe outputs |

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `android/app/build.gradle.kts` | Add `tensorflow-lite`, `tensorflow-lite-select-tf-ops` deps |
| 2 | `pubspec.yaml` | Add `tflite_flutter`, `tflite_flutter_helper` |
| 3 | `lib/pages/translate_page.dart` | Integrate `SignLanguageEngine` + display widget |

## Assets

| # | File | Source |
|---|------|--------|
| 1 | `assets/models/model_raw_fp16.tflite` | `model/` pipeline output |

## Dependencies

```yaml
# pubspec.yaml
dependencies:
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1
```

```kotlin
// android/app/build.gradle.kts
dependencies {
    // TFLite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")
}
```

## MethodChannel API (if using native Android for MediaPipe)

If MediaPipe runs on the native side (PlatformView), landmarks are
streamed to Dart via MethodChannel:

### Native → Dart

| Event | Payload | When |
|-------|---------|------|
| `onLandmarks` | `{pose: [[x,y,z]×33], left_hand: [[x,y,z]×21] or null, right_hand: [[x,y,z]×21] or null}` | Every frame |

### Dart → Native

| Call | Args | Returns | Notes |
|------|------|---------|-------|
| `startCamera` | – | `bool` | Start camera + MediaPipe |
| `stopCamera` | – | `bool` | Release resources |

## Implementation Order

1. Add `tflite_flutter` to `pubspec.yaml` + Flex delegate to `build.gradle.kts`
2. Download `model_raw_fp16.tflite` to `assets/models/`
3. Create `landmark_buffer.dart` — ring buffer with NaN-rotation
4. Create `tflite_interpreter.dart` — load model, verify shapes, run inference
5. Create `prediction_processor.dart` — softmax + temporal smoothing
6. Create `landmark_assembler.dart` — build 153-float frame from MediaPipe outputs
7. Create `sign_language_engine.dart` — orchestrate buffer → inference → smoothing
8. Integrate into `translate_page.dart` — start/stop engine with camera, display predictions
9. Test with pre-recorded landmarks (use `try.py` as a reference)
10. **Verify pose landmark indices** — compare predictions with `try.py`
11. Tune confidence threshold and smoothing window on-device

## Future Improvements

| Idea | Description |
|------|-------------|
| **On-device adaptation** | Adapt normalization stats to the user's hand/body shape |
| **Vocabulary expansion** | Swap model file to support more classes |
| **Real-time confidence display** | Show per-class confidence bars |
| **Prediction history** | Log predicted words in a session timeline |
| **Model download** | Fetch model from server on first launch instead of bundling |
