#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD_ROOT=${S2S_AEC3_BUILD_ROOT:-"$ROOT/.native-build/aec3"}
SOURCE_DIR="$BUILD_ROOT/webrtc-audio-processing"
INSTALL_DIR="$BUILD_ROOT/install"
MESON_BUILD_DIR="$BUILD_ROOT/meson-build"
LIBRARY="$ROOT/native/aec3/build/libs2s_aec3.dylib"

mkdir -p "$BUILD_ROOT" "$ROOT/native/aec3/build"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/okarlsen/webrtc-audio-processing.git "$SOURCE_DIR"
fi

meson setup "$MESON_BUILD_DIR" "$SOURCE_DIR" --prefix "$INSTALL_DIR" --buildtype=release --wipe
ninja -C "$MESON_BUILD_DIR"
ninja -C "$MESON_BUILD_DIR" install

if ! pkg-config --exists --define-prefix webrtc-audio-processing-2; then
  export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$INSTALL_DIR/lib/aarch64-apple-darwin/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

clang++ -std=c++17 -O3 -fPIC -dynamiclib \
  -I"$INSTALL_DIR/include" \
  native/aec3/aec3_wrapper.cc \
  $(pkg-config --cflags --libs webrtc-audio-processing-2) \
  -Wl,-rpath,@loader_path/../../../.native-build/aec3/install/lib \
  -o "$LIBRARY"

WEBRTC_LIBRARY="$INSTALL_DIR/lib/libwebrtc-audio-processing-2.1.dylib"
install_name_tool -id @rpath/libwebrtc-audio-processing-2.1.dylib "$WEBRTC_LIBRARY"
install_name_tool -change "$WEBRTC_LIBRARY" @rpath/libwebrtc-audio-processing-2.1.dylib "$LIBRARY"

echo "Built real WebRTC EchoCanceller3: $LIBRARY"
