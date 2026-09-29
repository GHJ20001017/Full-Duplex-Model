#!/usr/bin/env bash
set -euo pipefail

# ===== 本地配置：替换占位值；真实密钥不得提交到仓库 =====
# 已设置的环境变量优先于这里的默认值。
export OPENAI_API_KEY="${OPENAI_API_KEY:-替换为你的 API 密钥}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://你的服务域名/v1}"
export LLM_MODEL="${LLM_MODEL:-替换为该服务实际提供的模型名称}"
# 独立语义路由服务的完整接口 URL；按实际协议和地址修改。
# 同机且未配置 TLS 时使用 http://127.0.0.1:8792/v1/systemone。
export S2S_SEMANTIC_TURN_URL="${S2S_SEMANTIC_TURN_URL:-https://0.0.0.0:8792/v1/systemone}"
export S2S_SEMANTIC_TURN_TIMEOUT_S="${S2S_SEMANTIC_TURN_TIMEOUT_S:-1}" # 秒

# Nano accepts a local model directory here or via --fun_asr_nano_stt_model_name.
FUN_ASR_NANO_MODEL_DIR="${FUN_ASR_NANO_MODEL_DIR:-}"

# Keep the caller's working directory so relative model/hotword paths remain valid.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  printf '%s\n' \
    'Usage: bash scripts/start_voice_server.sh [serve options]' \
    'Default: Paraformer (CPU) + Chat Completions + Qwen3-TTS (CUDA/torch).' \
    'Configure OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL at the top of this script or via environment.' \
    'LLM_BASE_URL and LLM_MODEL may instead be supplied via their CLI options.' \
    'Start the semantic route service separately and set S2S_SEMANTIC_TURN_URL.' \
    'Nano: --stt fun-asr-nano (defaults to the project hotwords.txt).' \
    'Local Nano model: set FUN_ASR_NANO_MODEL_DIR=/path/to/model or pass --fun_asr_nano_stt_model_name /path/to/model.' \
    'Override hotwords: --fun_asr_nano_stt_hotwords_file /path/hotwords.txt' \
    'Extra serve options override defaults; both --option value and --option=value work.'
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

# Inspect only options needed for backend selection and preflight validation.
stt=paraformer
hotwords_file="$project_dir/hotwords.txt"
nano_model="${FUN_ASR_NANO_MODEL_DIR:-FunAudioLLM/Fun-ASR-Nano-2512}"
nano_local_dir="$FUN_ASR_NANO_MODEL_DIR"
base_url="${LLM_BASE_URL:-}"
model="${LLM_MODEL:-}"
forward=("$@")
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --stt|--fun_asr_nano_stt_model_name|--fun_asr_nano_stt_hotwords_file|--responses_api_base_url|--model_name)
      option="$1"
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "$option requires a value"
      value="$2"
      shift
      ;;
    --stt=*|--fun_asr_nano_stt_model_name=*|--fun_asr_nano_stt_hotwords_file=*|--responses_api_base_url=*|--model_name=*)
      option="${1%%=*}"
      value="${1#*=}"
      [[ -n "$value" ]] || fail "$option requires a value"
      ;;
    *) shift; continue ;;
  esac
  case "$option" in
    --stt) stt="$value" ;;
    --fun_asr_nano_stt_model_name) nano_model="$value"; nano_local_dir= ;;
    --fun_asr_nano_stt_hotwords_file) hotwords_file="$value" ;;
    --responses_api_base_url) base_url="$value" ;;
    --model_name) model="$value" ;;
  esac
  shift
done

[[ -n "$OPENAI_API_KEY" && "$OPENAI_API_KEY" != '替换为你的 API 密钥' ]] || fail 'Configure OPENAI_API_KEY in the script or environment.'
[[ -n "$base_url" && "$base_url" != 'https://你的服务域名/v1' ]] || fail 'Configure LLM_BASE_URL or pass --responses_api_base_url.'
[[ -n "$model" && "$model" != '替换为该服务实际提供的模型名称' ]] || fail 'Configure LLM_MODEL or pass --model_name.'

# Service and live transcription defaults; later user options take precedence.
args=(
  --host 0.0.0.0 --port 7869 --num_pipelines 1
  --interruption-route semantic
  --stt "$stt"
  --enable_live_transcription true
  --live_transcription_update_interval 0.5
)
case "$stt" in
  paraformer)
    args+=(--paraformer_stt_model_name paraformer-zh-streaming --paraformer_stt_device cpu)
    ;;
  fun-asr-nano)
    [[ -n "$hotwords_file" ]] || fail 'Nano requires --fun_asr_nano_stt_hotwords_file.'
    # Match the handler's expansion of a quoted ~/path.
    case "$hotwords_file" in '~/'*) hotwords_file="$HOME/${hotwords_file:2}" ;; esac
    [[ -f "$hotwords_file" && -r "$hotwords_file" && -s "$hotwords_file" ]] || fail 'Nano hotwords file must be a readable, nonempty regular file.'
    # UTF-8 parsing, comment filtering and phrase validation belong to the handler.
    case "$nano_model" in '~/'*) nano_model="$HOME/${nano_model:2}" ;; esac
    # Explicit filesystem paths must not silently become remote model IDs.
    case "$nano_model" in /*|./*|../*) nano_local_dir="$nano_model" ;; esac
    if [[ -n "$nano_local_dir" || -e "$nano_model" ]]; then
      [[ -d "$nano_model" && -r "$nano_model" && -x "$nano_model" ]] || fail "Nano model directory is not readable: $nano_model"
      nano_model="$(cd -- "$nano_model" && pwd)"
    fi
    args+=(--fun_asr_nano_stt_device cuda)
    # Append normalized paths after forwarded options so quoted ~/paths work too.
    forward+=(--fun_asr_nano_stt_model_name "$nano_model" --fun_asr_nano_stt_hotwords_file "$hotwords_file")
    ;;
esac

# LLM: credentials stay in the environment, never in the argument list.
args+=(
  --llm_backend chat-completions
  --responses_api_base_url "$base_url" --model_name "$model"
  --max_output_tokens 1024 --chat_size 30 --stream_batch_sentences 1
  --init_chat_role system
  --init_chat_prompt '你是 ARVIS，一个实时语音助手。请用简洁、自然的中文回答。'
  # TTS defaults require an NVIDIA GPU; they are not macOS defaults.
  --tts qwen3
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  --qwen3_tts_device cuda --qwen3_tts_backend torch
  --qwen3_tts_language Chinese --qwen3_tts_speaker Aiden
  --qwen3_tts_streaming_chunk_size 4 --qwen3_tts_max_new_tokens 1536
)

# Prefer the activated environment; fall back to the checkout's virtualenv.
if command -v speech-to-speech >/dev/null 2>&1; then
  executable="$(command -v speech-to-speech)"
elif [[ -x "$project_dir/.venv/bin/speech-to-speech" ]]; then
  executable="$project_dir/.venv/bin/speech-to-speech"
else
  fail 'speech-to-speech not found. Activate the installed environment first.'
fi
# Bash 3.2 treats an empty array as unset under nounset.
exec "$executable" serve "${args[@]}" ${forward[@]+"${forward[@]}"}
