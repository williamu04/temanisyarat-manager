# spec-005: Integrate MediaPipe + Flutter (Hand Landmarker)

## Overview

Integrate MediaPipe Hand Landmarker into the Teman Isyarat Flutter app
using a native Android `PlatformView` for real-time camera + hand
tracking, bridged via `AndroidView` widget in Flutter.

## Architecture

```
Flutter (Dart)
  ┌───────────────┐   ┌──────────────────────────┐
  │ TranslatePage │   │ AndroidView              │
  │ (Dart UI)     │──▶│ (PlatformView)           │
  │ + result box  │   │ ┌────────────────────┐   │
  │ + nav         │   │ │ MethodChannel      │   │
  └───────────────┘   │ └──────┬─────────────┘   │
                      └────────┼─────────────────┘
                               │
Native Android (Kotlin)        │
  ┌────────────────────────────▼──────────────────┐
  │    HandLandmarkerView (PlatformView)          │
  │  ┌────────────────┐   ┌────────────────────┐  │
  │  │ CameraX        │   │ HandLandmarker     │  │
  │  │ (PreviewView)  │   │ OverlayView        │  │
  │  └───────┬────────┘   │ (canvas skeleton)  │  │
  │          │            └────────────────────┘  │
  │  ┌───────▼──────────────────────────────────┐ │
  │  │ HandLandmarkerHelper (MediaPipe Tasks)   │ │
  │  │  - init / setup                          │ │
  │  │  - detectLiveStream (LIVE_STREAM mode)   │ │
  │  │  - callback -> onResults / onError       │ │
  │  └──────────────────────────────────────────┘ │
  │  ┌──────────────────────────────────────────┐ │
  │  │ GestureClassifier (future)               │ │
  │  │  - receives HandLandmarkerResult         │ │
  │  │  - returns gesture label + confidence    │ │
  │  └──────────────────────────────────────────┘ │
  └───────────────────────────────────────────────┘
```

## Files to Create

| #   | File                                                                                    | Purpose                                                                          |
| --- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1   | `android/app/src/main/java/com/example/android/handlandmarker/HandLandmarkerHelper.kt`  | MediaPipe Tasks wrapper - init, config, detectLiveStream, result/error listeners |
| 2   | `android/app/src/main/java/com/example/android/handlandmarker/HandLandmarkerView.kt`    | PlatformView - CameraX + PreviewView + overlay + lifecycle                       |
| 3   | `android/app/src/main/java/com/example/android/handlandmarker/HandLandmarkerOverlay.kt` | Custom View - draws hand skeleton (21 landmarks + HAND_CONNECTIONS)              |
| 4   | `android/app/src/main/java/com/example/android/handlandmarker/HandLandmarkerPlugin.kt`  | PlatformViewFactory + MethodChannel handler                                      |
| 5   | `lib/pages/translate_page.dart`                                                         | TranslatePage refactored out of main.dart, uses AndroidView                      |

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `android/app/build.gradle.kts` | Add `com.google.mediapipe:tasks-vision:0.10.29`, CameraX deps, set `minSdk = 24` |
| 2 | `android/app/src/main/AndroidManifest.xml` | Add `<uses-permission android:name="android.permission.CAMERA" />` |
| 3 | `lib/main.dart` | Replace TranslatePage placeholder with real AndroidView widget |

## Assets

| # | File | Source |
|---|------|--------|
| 1 | `android/app/src/main/assets/hand_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` |

## MethodChannel API

### Dart -> Native

| Call | Args | Returns | Notes |
|------|------|---------|-------|
| `startCamera` | - | `bool` | Initializes CameraX + MediaPipe |
| `stopCamera` | - | `bool` | Releases resources |
| `switchCamera` | - | `bool` | Front <-> back |
| `updateSettings` | `{maxHands: int, detectionConfidence: float, trackingConfidence: float, delegate: int}` | `bool` | 0=CPU, 1=GPU |

### Native -> Dart (via callback channel)

| Event | Data | Description |
|-------|------|-------------|
| `onGestureResult` | `{gesture: String, confidence: float}` | Gesture classification (future) |
| `onError` | `{message: String}` | Error reporting |
| `onLandmarks` | `{landmarks: [[x,y,z]]}` | Raw landmark positions |

## Gradle Dependencies

```kotlin
// MediaPipe Tasks Vision
implementation("com.google.mediapipe:tasks-vision:0.10.29")

// CameraX
val cameraxVersion = "1.4.2"
implementation("androidx.camera:camera-core:$cameraxVersion")
implementation("androidx.camera:camera-camera2:$cameraxVersion")
implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
implementation("androidx.camera:camera-view:$cameraxVersion")
```

## Implementation Order

1. Add Gradle dependencies + CAMERA permission
2. Download `hand_landmarker.task` to `android/app/src/main/assets/`
3. Create `HandLandmarkerHelper.kt` - adapted from Google's example
4. Create `HandLandmarkerOverlay.kt` - Canvas landmark skeleton
5. Create `HandLandmarkerView.kt` - PlatformView with CameraX
6. Create `HandLandmarkerPlugin.kt` - Factory + MethodChannel
7. Register plugin in MainActivity
8. Refactor TranslatePage + AndroidView into Dart
9. Build & verify

## Future: Gesture Classification

`HandLandmarkerHelper` includes a pluggable callback:

```kotlin
fun interface GestureClassifier {
    fun classify(landmarks: HandLandmarkerResult): Pair<String, Float>
}
```

When the classifier is set, `onResults` runs it and sends `onGestureResult` to Dart.
Classifier implementation (e.g. TF Lite model) can be added later without touching CameraX or MediaPipe pipeline.
