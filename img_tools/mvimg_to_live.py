#!/usr/bin/env python3
"""
mvimg_to_live.py
~~~~~~~~~~~~~~~~
从安卓 MVIMG 动态照片中批量提取内嵌 MP4 短视频，输出到指定目录。

用法：
    uv run python img_tools/mvimg_to_live.py [源目录] [--out 输出目录] [--workers N]

示例：
    uv run python img_tools/mvimg_to_live.py /Volumes/Share/等待导入照片 --out ~/导出
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

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

EXIFTOOL = shutil.which("exiftool") or "/opt/homebrew/bin/exiftool"

app = typer.Typer(
    name="mvimg-to-live",
    help="从安卓 MVIMG 动态照片批量提取内嵌 MP4 视频",
    no_args_is_help=True,
    add_completion=False,
)


# ─── 读取 XMP:MicroVideoOffset ───────────────────────────────────────────────

def get_video_offset(path: Path) -> int | None:
    """通过 exiftool 读取内嵌 MP4 从文件末尾的字节偏移量。"""
    result = subprocess.run(
        [EXIFTOOL, "-j", "-MicroVideoOffset", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        offset = data[0].get("MicroVideoOffset")
        return int(offset) if offset else None
    except (json.JSONDecodeError, IndexError, TypeError):
        return None


def find_mp4_by_magic(data: bytes) -> int | None:
    """扫描 ftyp box 定位内嵌 MP4（XMP offset 不可用时的降级方案）。"""
    pos = 0
    while True:
        idx = data.find(b"ftyp", pos)
        if idx < 4:
            if idx == -1:
                break
            pos = idx + 1
            continue
        box_size = struct.unpack(">I", data[idx - 4: idx])[0]
        if 16 <= box_size <= 512:
            return idx - 4
        pos = idx + 1
    return None


# ─── 单文件处理 ───────────────────────────────────────────────────────────────

def extract_mp4(mvimg_path: Path, out_dir: Path) -> bool:
    """从 MVIMG 文件提取内嵌 MP4，输出文件名保持与原文件相同（.mp4）。"""
    out_mp4 = out_dir / (mvimg_path.stem + ".mp4")

    if out_mp4.exists():
        log.info("已存在，跳过", file=out_mp4.name)
        return True

    raw = mvimg_path.read_bytes()
    file_size = len(raw)

    # 优先用 XMP offset
    offset = get_video_offset(mvimg_path)
    if offset and 0 < offset < file_size:
        mp4_start = file_size - offset
        if raw[mp4_start + 4: mp4_start + 8] == b"ftyp":
            out_mp4.write_bytes(raw[mp4_start:])
            src_stat = mvimg_path.stat()
            os.utime(out_mp4, (src_stat.st_atime, src_stat.st_mtime))
            log.info("✓ 提取成功", src=mvimg_path.name, dst=out_mp4.name,
                     size_mb=round(offset / 1e6, 1))
            return True

    # 降级：magic 扫描
    mp4_start = find_mp4_by_magic(raw)
    if mp4_start is not None and raw[:2] == b"\xff\xd8":
        mp4_data = raw[mp4_start:]
        out_mp4.write_bytes(mp4_data)
        src_stat = mvimg_path.stat()
        os.utime(out_mp4, (src_stat.st_atime, src_stat.st_mtime))
        log.info("✓ 提取成功（magic扫描）", src=mvimg_path.name, dst=out_mp4.name,
                 size_mb=round(len(mp4_data) / 1e6, 1))
        return True

    log.warning("✗ 无内嵌视频，跳过", file=mvimg_path.name)
    return False


# ─── 扫描 MVIMG 文件 ──────────────────────────────────────────────────────────

def find_mvimg_files(src_dir: Path) -> list[Path]:
    """递归扫描，找出所有含内嵌视频的 MVIMG 文件。"""
    results: list[Path] = []
    all_jpgs: list[Path] = []

    for f in sorted(src_dir.rglob("*")):
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        if name_lower.startswith("mvimg_") and name_lower.endswith((".jpg", ".jpeg")):
            results.append(f)
        elif name_lower.endswith(".mp.jpg"):  # Pixel 新格式
            results.append(f)
        elif name_lower.endswith((".jpg", ".jpeg")):
            all_jpgs.append(f)

    if all_jpgs:
        log.info("检测普通 JPEG 是否含动态照片标记", count=len(all_jpgs))
        r = subprocess.run(
            [EXIFTOOL, "-j", "-q", "-MicroVideo", "-MicroVideoOffset"]
            + [str(f) for f in all_jpgs],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            try:
                for entry in json.loads(r.stdout):
                    mv = entry.get("MicroVideo")
                    off = entry.get("MicroVideoOffset")
                    if (mv and str(mv) != "0") or (off and int(off) > 0):
                        results.append(Path(entry["SourceFile"]))
            except (json.JSONDecodeError, ValueError):
                pass

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    src: Annotated[
        Path,
        typer.Argument(help="源目录（含 MVIMG 动态照片）"),
    ] = Path("/Volumes/Share/等待导入照片"),
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="输出目录"),
    ] = Path.home() / "导出",
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="并发线程数"),
    ] = 2,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="仅列出待处理文件，不执行提取"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="输出调试日志"),
    ] = False,
) -> None:
    """从安卓 MVIMG 动态照片批量提取内嵌 MP4 视频。"""
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10)
        )

    if not Path(EXIFTOOL).exists():
        log.error("缺少 exiftool，请运行：brew install exiftool")
        raise typer.Exit(1)

    src_dir = src.expanduser().resolve()
    out_dir = out.expanduser().resolve()

    if not src_dir.exists():
        log.error("源目录不存在", path=str(src_dir))
        raise typer.Exit(1)

    log.info("开始处理", src=str(src_dir), out=str(out_dir))

    files = find_mvimg_files(src_dir)
    log.info("找到 MVIMG 动态照片", count=len(files))

    if not files:
        log.info("没有找到需要处理的文件")
        return

    if dry_run:
        for f in files:
            typer.echo(str(f))
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    success = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_mp4, f, out_dir): f for f in files}
        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                log.error("处理出错", file=futures[future].name, error=str(e))
                fail += 1

    log.info("全部完成", success=success, failed=fail, out=str(out_dir))


if __name__ == "__main__":
    app()
