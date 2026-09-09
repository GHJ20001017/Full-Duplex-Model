#include <cstdint>
#include <memory>
#include <mutex>

#include "api/audio/audio_processing.h"

namespace {

class Aec3Processor {
 public:
  explicit Aec3Processor(int sample_rate) : sample_rate_(sample_rate) {
    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = true;
    config.echo_canceller.mobile_mode = false;
    config.high_pass_filter.enabled = true;
    apm_ = webrtc::AudioProcessingBuilder().SetConfig(config).Create();
    stream_config_ = std::make_unique<webrtc::StreamConfig>(sample_rate_, 1);
  }

  int ProcessRender(const int16_t* audio, int samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    return apm_->ProcessReverseStream(audio, *stream_config_, *stream_config_,
                                       const_cast<int16_t*>(audio));
  }

  int ProcessCapture(const int16_t* input, int16_t* output, int samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    return apm_->ProcessStream(input, *stream_config_, *stream_config_, output);
  }

 private:
  int sample_rate_;
  rtc::scoped_refptr<webrtc::AudioProcessing> apm_;
  std::unique_ptr<webrtc::StreamConfig> stream_config_;
  std::mutex mutex_;
};

}  // namespace

extern "C" {

void* s2s_aec3_create(int sample_rate) {
  if (sample_rate <= 0) return nullptr;
  try {
    return new Aec3Processor(sample_rate);
  } catch (...) {
    return nullptr;
  }
}

int s2s_aec3_process_render(void* handle, const int16_t* audio, int samples) {
  if (!handle || !audio || samples <= 0) return -1;
  return static_cast<Aec3Processor*>(handle)->ProcessRender(audio, samples);
}

int s2s_aec3_process_capture(void* handle, const int16_t* input,
                             int16_t* output, int samples) {
  if (!handle || !input || !output || samples <= 0) return -1;
  return static_cast<Aec3Processor*>(handle)->ProcessCapture(input, output, samples);
}

void s2s_aec3_destroy(void* handle) {
  delete static_cast<Aec3Processor*>(handle);
}

}
