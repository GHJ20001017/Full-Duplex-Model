"""Download pretrained Qwen2.5-Omni assets from ModelScope into ../model/."""
import argparse
from pathlib import Path
import sys


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MODEL_IDS = (MODEL_ID, "Qwen/Qwen2.5-Omni-3B")
MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
MODEL_DIR = MODEL_ROOT / MODEL_ID.rsplit("/", 1)[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载 Qwen2.5-Omni 的完整模型文件（包含语音生成组件）。",
        epilog=f"默认保存目录：{MODEL_DIR}。下载中断后可重新运行，复用已缓存文件。",
    )
    parser.add_argument(
        "--model-id", choices=MODEL_IDS, default=MODEL_ID,
        help="官方模型 ID，默认 %(default)s。",
    )
    parser.add_argument(
        "--local-dir", type=Path,
        help="保存目录，默认使用本项目 model/<模型名称>。",
    )
    args = parser.parse_args(argv)
    model_dir = (
        args.local_dir.expanduser().resolve()
        if args.local_dir is not None
        else MODEL_ROOT / args.model_id.rsplit("/", 1)[-1]
    )

    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError as exc:
        if exc.name != "modelscope":
            raise
        print(
            f"缺少 modelscope，请先执行：{sys.executable} -m pip install modelscope",
            file=sys.stderr,
        )
        return 1

    print(f"模型：{args.model_id}", flush=True)
    print(f"保存目录：{model_dir}", flush=True)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(args.model_id, local_dir=str(model_dir))
    except KeyboardInterrupt:
        print("\n下载已中断，可重新运行脚本继续下载。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"下载失败：{exc}\n修复后可重新运行脚本。", file=sys.stderr)
        return 1

    print(f"模型下载完成：{model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
