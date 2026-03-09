#!/usr/bin/env python3
"""
filter_egomotion.py
~~~~~~~~~~~~~~~~~~~
从导出的动态视频中检测相机 Ego-motion（剧烈自我运动，如"放下手机"），
将视频分类到两个文件夹，并可选地裁剪掉末尾的剧烈运动片段。

用法：
    uv run python img_tools/filter_egomotion.py <输入目录> [选项]

示例：
    # 仅分类（有/无剧烈运动）
    uv run python img_tools/filter_egomotion.py 导出动态视频 --task classify

    # 仅裁剪（去除末尾ego-motion）
    uv run python img_tools/filter_egomotion.py 导出动态视频 --task trim

    # 全部执行（分类 + 裁剪）
    uv run python img_tools/filter_egomotion.py 导出动态视频 --task all
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, NamedTuple, Optional

import cv2
import numpy as np
import structlog
import typer

# ─── 日志配置 ─────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)
log = structlog.get_logger()

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

app = typer.Typer(
    name="filter-egomotion",
    help="检测并处理视频中的相机 Ego-motion（剧烈自我运动）",
    no_args_is_help=True,
    add_completion=False,
)


# ─── 类型定义 ──────────────────────────────────────────────────────────────────

class Task(str, Enum):
    classify = "classify"
    trim = "trim"
    all = "all"


class EgoMotionResult(NamedTuple):
    has_egomotion: bool
    cut_time: float | None
    max_flow: float
    motion_segments: list[tuple[float, float]]


# ─── 光流检测核心 ──────────────────────────────────────────────────────────────

def compute_global_flow(frame1_gray: np.ndarray, frame2_gray: np.ndarray) -> float:
    """计算两帧之间的全局运动强度（稠密光流中值）。"""
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.median(magnitude))


def detect_egomotion(
    video_path: Path,
    threshold: float = 8.0,
    min_duration: float = 0.4,
    fps_sample: int = 6,
    full_scan: bool = True,
    tail_window: float = 5.0,
    resize_width: int = 320,
) -> EgoMotionResult:
    """对单个视频进行 ego-motion 检测。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("无法打开视频", file=video_path.name)
        return EgoMotionResult(False, None, 0.0, [])

    total_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / total_fps
    frame_interval = max(1, int(total_fps / fps_sample))

    if full_scan:
        start_frame = 0
    else:
        start_sec = max(0.0, total_duration - tail_window)
        start_frame = int(start_sec * total_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray: np.ndarray | None = None
    frame_idx = start_frame
    timestamps: list[float] = []
    flows: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = frame_idx / total_fps
        if prev_gray is not None:
            flow_mag = compute_global_flow(prev_gray, gray)
            timestamps.append(t)
            flows.append(flow_mag)
        prev_gray = gray
        frame_idx += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()

    if not flows:
        return EgoMotionResult(False, None, 0.0, [])

    max_flow = max(flows)
    motion_segments: list[tuple[float, float]] = []
    seg_start: float | None = None
    dt = 1.0 / fps_sample

    for i, (t, f) in enumerate(zip(timestamps, flows)):
        if f >= threshold:
            if seg_start is None:
                seg_start = t
        else:
            if seg_start is not None:
                seg_end = timestamps[i - 1] if i > 0 else t
                duration = seg_end - seg_start + dt
                if duration >= min_duration:
                    motion_segments.append((seg_start, seg_end + dt))
                seg_start = None

    if seg_start is not None:
        seg_end = timestamps[-1]
        duration = seg_end - seg_start + dt
        if duration >= min_duration:
            motion_segments.append((seg_start, seg_end + dt))

    has_egomotion = len(motion_segments) > 0
    cut_time: float | None = None
    if has_egomotion:
        cut_time = motion_segments[0][0]
        if cut_time < 0.5:
            cut_time = 0.0

    return EgoMotionResult(has_egomotion, cut_time, max_flow, motion_segments)


# ─── 任务1：分类 ───────────────────────────────────────────────────────────────

def classify_video(
    video_path: Path,
    out_clean: Path,
    out_noisy: Path,
    threshold: float,
    min_duration: float,
    fps_sample: int,
    dry_run: bool,
) -> tuple[Path, EgoMotionResult]:
    """对单个视频分类，复制到对应目录。"""
    result = detect_egomotion(
        video_path, threshold=threshold, min_duration=min_duration,
        fps_sample=fps_sample, full_scan=True,
    )
    target_dir = out_noisy if result.has_egomotion else out_clean
    dest = target_dir / video_path.name
    label = "有剧烈运动" if result.has_egomotion else "无剧烈运动"

    log.info(
        f"[{label}]",
        file=video_path.name,
        max_flow=round(result.max_flow, 1),
        segments=result.motion_segments or None,
    )

    if not dry_run:
        if dest.exists():
            log.debug("已存在，跳过复制", file=dest.name)
        else:
            try:
                os.link(video_path, dest)
            except OSError:
                shutil.copy2(video_path, dest)

    return video_path, result


# ─── 任务2：裁剪 ───────────────────────────────────────────────────────────────

def trim_video(
    video_path: Path,
    out_trimmed: Path,
    threshold: float,
    min_duration: float,
    fps_sample: int,
    tail_window: float,
    dry_run: bool,
) -> tuple[Path, bool]:
    """检测末尾 ego-motion 并裁剪。"""
    result = detect_egomotion(
        video_path, threshold=threshold, min_duration=min_duration,
        fps_sample=fps_sample, full_scan=False, tail_window=tail_window,
    )

    if not result.has_egomotion:
        log.info("[跳过]", file=video_path.name, max_flow=round(result.max_flow, 1),
                 reason="无末尾运动")
        return video_path, False

    cut_time = result.cut_time
    if cut_time is None or cut_time <= 0.5:
        log.warning("[跳过]", file=video_path.name, cut_time=cut_time,
                    reason="裁剪点过小，整段可能都是运动")
        return video_path, False

    dest = out_trimmed / video_path.name
    log.info("[裁剪]", file=video_path.name, cut_at_sec=round(cut_time, 2),
             max_flow=round(result.max_flow, 1), segments=result.motion_segments)

    if not dry_run:
        if dest.exists():
            log.debug("已存在，跳过", file=dest.name)
        else:
            _ffmpeg_trim(video_path, dest, cut_time)

    return video_path, True


def _ffmpeg_trim(src: Path, dst: Path, duration: float) -> None:
    """用 ffmpeg 无损截取视频前 duration 秒。"""
    if not Path(FFMPEG).exists():
        raise RuntimeError(f"找不到 ffmpeg：{FFMPEG}，请运行 brew install ffmpeg")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{result.stderr.strip()}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    src: Annotated[
        Path,
        typer.Argument(help="输入目录（包含 MP4 文件）"),
    ] = Path("导出动态视频"),
    task: Annotated[
        Task,
        typer.Option("--task", "-t", help="执行任务：classify=仅分类，trim=仅裁剪，all=两者"),
    ] = Task.all,
    out_clean: Annotated[
        Optional[Path],
        typer.Option("--out-clean", help="无 ego-motion 视频输出目录（默认：<src>/../无剧烈运动）"),
    ] = None,
    out_noisy: Annotated[
        Optional[Path],
        typer.Option("--out-noisy", help="有 ego-motion 视频输出目录（默认：<src>/../有剧烈运动）"),
    ] = None,
    out_trimmed: Annotated[
        Optional[Path],
        typer.Option("--out-trimmed", help="裁剪后视频输出目录（默认：<src>/../裁剪后）"),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="全局光流幅度阈值（像素/帧）"),
    ] = 8.0,
    min_duration: Annotated[
        float,
        typer.Option("--min-duration", help="触发 ego-motion 所需的最短连续运动时长（秒）"),
    ] = 0.4,
    fps_sample: Annotated[
        int,
        typer.Option("--fps-sample", min=1, help="每秒采样帧数"),
    ] = 6,
    tail_window: Annotated[
        float,
        typer.Option("--tail-window", help="裁剪任务：从末尾往前检测的秒数"),
    ] = 5.0,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="并发线程数"),
    ] = 4,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="仅分析，不复制/裁剪文件"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="输出调试日志"),
    ] = False,
) -> None:
    """检测并处理视频中的相机 Ego-motion（剧烈自我运动）。"""
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10)
        )

    src_dir = src.expanduser().resolve()
    if not src_dir.exists():
        log.error("输入目录不存在", path=str(src_dir))
        raise typer.Exit(1)

    parent = src_dir.parent
    clean_dir   = (out_clean.expanduser().resolve()   if out_clean   else parent / "无剧烈运动")
    noisy_dir   = (out_noisy.expanduser().resolve()   if out_noisy   else parent / "有剧烈运动")
    trimmed_dir = (out_trimmed.expanduser().resolve() if out_trimmed else parent / "裁剪后")

    do_classify = task in (Task.classify, Task.all)
    do_trim     = task in (Task.trim, Task.all)

    if not dry_run:
        if do_classify:
            clean_dir.mkdir(parents=True, exist_ok=True)
            noisy_dir.mkdir(parents=True, exist_ok=True)
        if do_trim:
            trimmed_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".mp4" and p.is_file())
    if not videos:
        log.info("没有找到 MP4 文件", path=str(src_dir))
        return

    log.info("找到视频文件", count=len(videos), threshold=threshold,
             fps_sample=fps_sample, workers=workers)
    if dry_run:
        log.info("【DRY RUN】不会写入任何文件")

    noisy_paths: list[Path] = []

    # ── Task 1: 分类 ────────────────────────────────────────────────────────────
    if do_classify:
        log.info("▶ 开始分类（全程扫描）")
        n_clean = n_noisy = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    classify_video, v, clean_dir, noisy_dir,
                    threshold, min_duration, fps_sample, dry_run,
                ): v
                for v in videos
            }
            for future in as_completed(futures):
                try:
                    path, result = future.result()
                    if result.has_egomotion:
                        noisy_paths.append(path)
                        n_noisy += 1
                    else:
                        n_clean += 1
                except Exception as e:
                    log.error("分类出错", file=futures[future].name, error=str(e))

        log.info("分类完成", clean=n_clean, noisy=n_noisy,
                 out_clean=str(clean_dir), out_noisy=str(noisy_dir))

    # ── Task 2: 裁剪 ────────────────────────────────────────────────────────────
    if do_trim:
        trim_targets = noisy_paths if do_classify else videos
        log.info("▶ 开始裁剪", tail_window=tail_window, targets=len(trim_targets))
        n_trimmed = n_skip = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    trim_video, v, trimmed_dir,
                    threshold, min_duration, fps_sample, tail_window, dry_run,
                ): v
                for v in trim_targets
            }
            for future in as_completed(futures):
                try:
                    path, trimmed = future.result()
                    if trimmed:
                        n_trimmed += 1
                    else:
                        n_skip += 1
                except Exception as e:
                    log.error("裁剪出错", file=futures[future].name, error=str(e))

        log.info("裁剪完成", trimmed=n_trimmed, skipped=n_skip, out=str(trimmed_dir))

    log.info("全部完成！")


if __name__ == "__main__":
    app()
