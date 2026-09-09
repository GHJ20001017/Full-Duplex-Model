# Native WebRTC AEC3

This directory contains the small C ABI adapter used by the local microphone
client. The build script vendors the modern `webrtc-audio-processing` 2.x
AudioProcessing module and enables the real WebRTC `EchoCanceller3` path:

```cpp
config.echo_canceller.enabled = true;
config.echo_canceller.mobile_mode = false;
```

Build on Apple Silicon macOS:

```bash
./native/aec3/build_macos.sh
```

The script produces `native/aec3/build/libs2s_aec3.dylib`. The generated
library and the vendored build tree are intentionally ignored by Git.

The client uses 10 ms mono PCM frames internally. Render audio is fed to
`ProcessReverseStream`, and microphone audio is fed to `ProcessStream` before
the wake-word gate and WebSocket upload.
