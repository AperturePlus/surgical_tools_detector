# Android Edge Prototype

This project keeps the desktop implementation as the production path and treats Android on-device inference as a separate feasibility track.

## Target Architecture

- Android UI: native Kotlin app with CameraX preview and a capture button.
- Inference runtime: ONNX Runtime Android from Maven Central.
- First execution providers to test: CPU/XNNPACK, then NNAPI only if CPU/XNNPACK misses the target FPS.
- Shared behavior with desktop: reuse the same model files, image preprocessing rules, detection thresholds, and capture JSON schema.
- Output parity check: run the same still image through desktop and Android and compare labels, bounding boxes, confidence scores, and defect scores.

## Gradle Dependency Sketch

```gradle
repositories {
    mavenCentral()
}

dependencies {
    implementation "com.microsoft.onnxruntime:onnxruntime-android:latest.release"
    implementation "androidx.camera:camera-camera2:latest.release"
    implementation "androidx.camera:camera-lifecycle:latest.release"
    implementation "androidx.camera:camera-view:latest.release"
}
```

## Acceptance Gates

- The three current ONNX models load on a representative arm64 Android device.
- A fixed test image produces comparable results to the desktop pipeline.
- The app records FPS, memory use, model load time, and capture latency.
- A capture writes raw image, annotated image, and schema version 1 JSON.

## Current Local Status

This workstation does not currently expose `gradle`, `ANDROID_HOME`, or `ANDROID_SDK_ROOT`, so the Android prototype has not been compiled locally. The desktop Qt implementation and capture storage path are build-verified in this repository.
