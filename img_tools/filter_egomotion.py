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

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

class EgoMotionResult(NamedTuple):
    has_egomotion: bool          # 是否存在 ego-motion
    cut_time: float | None       # 建议裁剪点（秒），None 表示全程无 ego-motion
    max_flow: float              # 全程最大光流幅度
    motion_segments: list[tuple[float, float]]  # [(start_sec, end_sec), ...]


# ─── 光流检测核心 ──────────────────────────────────────────────────────────────

def compute_global_flow(frame1_gray: np.ndarray, frame2_gray: np.ndarray) -> float:
    """计算两帧之间的全局运动强度（稠密光流中值）。"""
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # 使用中值而非均值，对噪声更鲁棒
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
    """
    对单个视频进行 ego-motion 检测。

    Args:
        video_path:    视频文件路径
        threshold:     全局光流幅度阈值（像素/帧），超过则视为运动帧
        min_duration:  连续运动帧持续时间（秒），达到后才算一段 ego-motion
        fps_sample:    每秒采样帧数
        full_scan:     True=全程扫描；False=仅检测末尾 tail_window 秒
        tail_window:   末尾检测窗口（秒），full_scan=False 时有效
        resize_width:  缩放宽度（加速计算）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("无法打开视频：%s", video_path.name)
        return EgoMotionResult(False, None, 0.0, [])

    total_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / total_fps

    # 计算采样间隔（帧数）
    frame_interval = max(1, int(total_fps / fps_sample))

    # 确定扫描起始帧
    if full_scan:
        start_frame = 0
    else:
        start_sec = max(0.0, total_duration - tail_window)
        start_frame = int(start_sec * total_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray: np.ndarray | None = None
    frame_idx = start_frame

    # 每个采样点的时间戳 + 光流幅度
    timestamps: list[float] = []
    flows: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩放以加速
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

        # 跳到下一个采样帧
        next_frame = frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    cap.release()

    if not flows:
        return EgoMotionResult(False, None, 0.0, [])

    max_flow = max(flows)

    # ── 检测连续超阈值片段 ──────────────────────────────────────────────────────
    motion_segments: list[tuple[float, float]] = []
    seg_start: float | None = None
    dt = 1.0 / fps_sample  # 两个采样点之间的时间间隔

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

    # 收尾：视频结束时仍在运动
    if seg_start is not None:
        seg_end = timestamps[-1]
        duration = seg_end - seg_start + dt
        if duration >= min_duration:
            motion_segments.append((seg_start, seg_end + dt))

    has_egomotion = len(motion_segments) > 0

    # 建议裁剪点：第一段 ego-motion 的起始时间（保留之前的画面）
    cut_time: float | None = None
    if has_egomotion:
        cut_time = motion_segments[0][0]
        # 避免裁剪点过小（若 ego-motion 在最开始就出现，保留 0 秒无意义）
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
        video_path,
        threshold=threshold,
        min_duration=min_duration,
        fps_sample=fps_sample,
        full_scan=True,  # 分类时全程扫描
    )

    target_dir = out_noisy if result.has_egomotion else out_clean
    dest = target_dir / video_path.name

    label = "有剧烈运动" if result.has_egomotion else "无剧烈运动"
    segs = f" | 片段: {result.motion_segments}" if result.motion_segments else ""
    log.info(
        "[%s] %s  (max_flow=%.1f%s)",
        label, video_path.name, result.max_flow, segs,
    )

    if not dry_run:
        if dest.exists():
            log.debug("已存在，跳过复制：%s", dest.name)
        else:
            try:
                os.link(video_path, dest)  # 尝试硬链接（不占额外空间）
            except OSError:
                shutil.copy2(video_path, dest)  # 跨设备则复制

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
    """
    对单个视频检测末尾 ego-motion，裁剪后输出。
    若无 ego-motion，跳过（不输出）。
    """
    result = detect_egomotion(
        video_path,
        threshold=threshold,
        min_duration=min_duration,
        fps_sample=fps_sample,
        full_scan=False,  # 裁剪任务只看末尾
        tail_window=tail_window,
    )

    if not result.has_egomotion:
        log.info("[跳过] %s  (无末尾运动, max_flow=%.1f)", video_path.name, result.max_flow)
        return video_path, False

    cut_time = result.cut_time
    if cut_time is None or cut_time <= 0.5:
        log.warning("[跳过] %s  (裁剪点 %.1fs 过小，整段可能都是运动)", video_path.name, cut_time or 0)
        return video_path, False

    dest = out_trimmed / video_path.name
    log.info(
        "[裁剪] %s  →  %.2fs 处截断  (max_flow=%.1f, 片段: %s)",
        video_path.name, cut_time, result.max_flow, result.motion_segments,
    )

    if not dry_run:
        if dest.exists():
            log.debug("已存在，跳过：%s", dest.name)
        else:
            _ffmpeg_trim(video_path, dest, cut_time)

    return video_path, True


def _ffmpeg_trim(src: Path, dst: Path, duration: float) -> None:
    """用 ffmpeg 无损截取视频前 duration 秒。"""
    if not Path(FFMPEG).exists():
        raise RuntimeError(f"找不到 ffmpeg：{FFMPEG}，请运行 brew install ffmpeg")

    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",          # 无损复制，速度快
        "-avoid_negative_ts", "make_zero",
        "-y",                  # 覆盖
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{result.stderr.strip()}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检测并处理视频中的相机 Ego-motion（剧烈自我运动）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="示例：uv run python img_tools/filter_egomotion.py 导出动态视频 --task all",
    )
    parser.add_argument(
        "src",
        nargs="?",
        default="导出动态视频",
        help="输入目录（包含 MP4 文件）",
    )
    parser.add_argument(
        "--task",
        choices=["classify", "trim", "all"],
        default="all",
        help="执行任务：classify=仅分类，trim=仅裁剪，all=分类+裁剪",
    )
    parser.add_argument(
        "--out-clean",
        default=None,
        metavar="DIR",
        help="无 ego-motion 视频输出目录（默认：<src>/../无剧烈运动）",
    )
    parser.add_argument(
        "--out-noisy",
        default=None,
        metavar="DIR",
        help="有 ego-motion 视频输出目录（默认：<src>/../有剧烈运动）",
    )
    parser.add_argument(
        "--out-trimmed",
        default=None,
        metavar="DIR",
        help="裁剪后视频输出目录（默认：<src>/../裁剪后）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        metavar="FLOAT",
        help="全局光流幅度阈值（像素/帧）",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.4,
        metavar="SEC",
        help="触发 ego-motion 所需的最短连续运动时长（秒）",
    )
    parser.add_argument(
        "--fps-sample",
        type=int,
        default=6,
        metavar="N",
        help="每秒采样帧数（越大越精确，越慢）",
    )
    parser.add_argument(
        "--tail-window",
        type=float,
        default=5.0,
        metavar="SEC",
        help="裁剪任务：从末尾往前检测的秒数",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="并发线程数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅分析，不复制/裁剪文件",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    if not src_dir.exists():
        log.error("输入目录不存在：%s", src_dir)
        sys.exit(1)

    parent = src_dir.parent

    out_clean   = Path(args.out_clean).expanduser().resolve()   if args.out_clean   else parent / "无剧烈运动"
    out_noisy   = Path(args.out_noisy).expanduser().resolve()   if args.out_noisy   else parent / "有剧烈运动"
    out_trimmed = Path(args.out_trimmed).expanduser().resolve() if args.out_trimmed else parent / "裁剪后"

    do_classify = args.task in ("classify", "all")
    do_trim     = args.task in ("trim", "all")

    if not args.dry_run:
        if do_classify:
            out_clean.mkdir(parents=True, exist_ok=True)
            out_noisy.mkdir(parents=True, exist_ok=True)
        if do_trim:
            out_trimmed.mkdir(parents=True, exist_ok=True)

    # 收集视频文件
    videos = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".mp4" and p.is_file())
    if not videos:
        log.info("没有找到 MP4 文件：%s", src_dir)
        return

    log.info("找到 %d 个视频，阈值=%.1f，采样=%d fps，并发=%d",
             len(videos), args.threshold, args.fps_sample, args.workers)
    if args.dry_run:
        log.info("【DRY RUN】不会写入任何文件")

    # ── Task 1: 分类 ────────────────────────────────────────────────────────────
    if do_classify:
        log.info("━" * 56)
        log.info("▶ 开始分类（全程扫描）")

        noisy_paths: list[Path] = []  # 记录有 ego-motion 的视频，供 trim 使用
        n_clean = n_noisy = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_video,
                    v, out_clean, out_noisy,
                    args.threshold, args.min_duration, args.fps_sample,
                    args.dry_run,
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
                    log.error("分类出错 %s：%s", futures[future].name, e)

        log.info("━" * 56)
        log.info("分类完成：无剧烈运动 %d 个 → %s", n_clean, out_clean)
        log.info("分类完成：有剧烈运动 %d 个 → %s", n_noisy, out_noisy)

    # ── Task 2: 裁剪 ────────────────────────────────────────────────────────────
    if do_trim:
        # 如果只运行 trim（未先分类），则扫描所有视频
        trim_targets = noisy_paths if do_classify else videos

        log.info("━" * 56)
        log.info("▶ 开始裁剪（检测末尾 %.1fs 内的 ego-motion）", args.tail_window)

        n_trimmed = n_skip = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    trim_video,
                    v, out_trimmed,
                    args.threshold, args.min_duration, args.fps_sample,
                    args.tail_window,
                    args.dry_run,
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
                    log.error("裁剪出错 %s：%s", futures[future].name, e)

        log.info("━" * 56)
        log.info("裁剪完成：已裁剪 %d 个，跳过 %d 个 → %s", n_trimmed, n_skip, out_trimmed)

    log.info("全部完成！")


if __name__ == "__main__":
    main()
