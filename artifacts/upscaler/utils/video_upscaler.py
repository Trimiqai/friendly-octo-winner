import cv2
import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

COMPRESSION_SETTINGS = {
    "high_quality": {"crf": 17, "preset": "slow"},
    "balanced":     {"crf": 23, "preset": "medium"},
    "small_size":   {"crf": 28, "preset": "fast"},
}


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=0",
            video_path,
        ],
        capture_output=True, text=True, timeout=30
    )
    info = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def extract_frames(video_path: str, frames_dir: str, progress_cb: Optional[Callable] = None) -> int:
    os.makedirs(frames_dir, exist_ok=True)
    logger.info(f"Extracting frames from {video_path} → {frames_dir}")

    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-q:v", "1",
            os.path.join(frames_dir, "frame_%06d.png"),
            "-y",
        ],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr}")

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    count = len(frames)
    logger.info(f"Extracted {count} frames")
    return count


def upscale_frames(
    frames_dir: str,
    upscaled_dir: str,
    scale: int,
    progress_cb: Optional[Callable] = None,
):
    from .models_manager import get_upscaler
    import cv2

    os.makedirs(upscaled_dir, exist_ok=True)
    upscaler = get_upscaler(scale)

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    total = len(frames)
    logger.info(f"Upscaling {total} frames at {scale}x...")

    for idx, frame_name in enumerate(frames, 1):
        src = os.path.join(frames_dir, frame_name)
        dst = os.path.join(upscaled_dir, frame_name)

        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not read frame {frame_name}, skipping.")
            continue

        upscaled, _ = upscaler.enhance(img, outscale=scale)
        cv2.imwrite(dst, upscaled)

        if idx % 10 == 0 or idx == total:
            pct = idx / total * 100
            logger.info(f"  Frame {idx}/{total} ({pct:.1f}%)")
            if progress_cb:
                progress_cb(idx, total)


def merge_frames_to_video(
    upscaled_dir: str,
    original_video: str,
    output_path: str,
    compression: str,
):
    settings = COMPRESSION_SETTINGS.get(compression.lower(), COMPRESSION_SETTINGS["balanced"])

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1",
         original_video],
        capture_output=True, text=True, timeout=30
    )
    fps_str = result.stdout.strip() or "25/1"

    logger.info(f"Merging frames with fps={fps_str}, compression={compression}")

    frame_pattern = os.path.join(upscaled_dir, "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-framerate", fps_str,
        "-i", frame_pattern,
        "-i", original_video,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", str(settings["crf"]),
        "-preset", settings["preset"],
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

    logger.info(f"Video merged → {output_path}")


def upscale_video(
    input_path: str,
    output_path: str,
    scale: int,
    compression: str,
    progress_cb: Optional[Callable] = None,
) -> str:
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    base = Path(input_path).stem
    work_dir = Path(input_path).parent / f"work_{base}"
    frames_dir = str(work_dir / "frames")
    upscaled_dir = str(work_dir / "upscaled")

    try:
        frame_count = extract_frames(input_path, frames_dir)
        if progress_cb:
            progress_cb("extracting_frames", frame_count)

        upscale_frames(frames_dir, upscaled_dir, scale, progress_cb=None)
        if progress_cb:
            progress_cb("merging_video", 0)

        merge_frames_to_video(upscaled_dir, input_path, output_path, compression)
        logger.info(f"Video upscaling complete: {output_path}")
        return output_path
    finally:
        if work_dir.exists():
            shutil.rmtree(str(work_dir), ignore_errors=True)
