"""Download VoxBase-omni model assets from ModelScope into ../model/."""
import argparse
from pathlib import Path
import sys


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "Qwen3-4B-Instruct-2507"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=f"从 ModelScope 下载 {MODEL_ID} 的完整模型文件。",
        epilog=f"保存目录：{MODEL_DIR}。下载中断后可重新运行，复用已缓存文件。",
    )
    parser.parse_args(argv)

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

    print(f"模型：{MODEL_ID}", flush=True)
    print(f"保存目录：{MODEL_DIR}", flush=True)
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
    except KeyboardInterrupt:
        print("\n下载已中断，可重新运行脚本继续下载。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"下载失败：{exc}\n修复后可重新运行脚本。", file=sys.stderr)
        return 1

    print(f"模型下载完成：{MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
