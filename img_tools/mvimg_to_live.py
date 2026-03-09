#!/usr/bin/env python3
"""
mvimg_to_mp4.py
~~~~~~~~~~~~~~~
从安卓 MVIMG 动态照片中批量提取内嵌 MP4 短视频，输出到指定目录。

用法：
    uv run python mvimg_to_live.py [源目录] [--out 输出目录] [--workers N]

示例：
    uv run python mvimg_to_live.py /Volumes/Share/等待导入照片 --out ~/导出
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXIFTOOL = shutil.which("exiftool") or "/opt/homebrew/bin/exiftool"


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
        log.info("已存在，跳过：%s", out_mp4.name)
        return True

    raw = mvimg_path.read_bytes()
    file_size = len(raw)

    # 优先用 XMP offset
    offset = get_video_offset(mvimg_path)
    if offset and 0 < offset < file_size:
        mp4_start = file_size - offset
        if raw[mp4_start + 4: mp4_start + 8] == b"ftyp":
            out_mp4.write_bytes(raw[mp4_start:])
            # 保留原始文件的修改时间（大致对应拍摄时间）
            src_stat = mvimg_path.stat()
            os.utime(out_mp4, (src_stat.st_atime, src_stat.st_mtime))
            log.info("✓ %s  →  %s  (%.1f MB)", mvimg_path.name, out_mp4.name, offset / 1e6)
            return True

    # 降级：magic 扫描
    mp4_start = find_mp4_by_magic(raw)
    if mp4_start is not None and raw[:2] == b"\xff\xd8":
        mp4_data = raw[mp4_start:]
        out_mp4.write_bytes(mp4_data)
        src_stat = mvimg_path.stat()
        os.utime(out_mp4, (src_stat.st_atime, src_stat.st_mtime))
        log.info("✓ %s  →  %s  (magic扫描, %.1f MB)", mvimg_path.name, out_mp4.name, len(mp4_data) / 1e6)
        return True

    log.warning("✗ 无内嵌视频，跳过：%s", mvimg_path.name)
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
        # 明确匹配 MVIMG 命名
        if name_lower.startswith("mvimg_") and name_lower.endswith((".jpg", ".jpeg")):
            results.append(f)
        elif name_lower.endswith(".mp.jpg"):  # Pixel 新格式
            results.append(f)
        elif name_lower.endswith((".jpg", ".jpeg")):
            all_jpgs.append(f)

    # 对剩余普通 JPEG 批量检测 XMP 标签
    if all_jpgs:
        log.info("检测剩余 %d 个 JPEG 是否含动态照片标记…", len(all_jpgs))
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="从安卓 MVIMG 动态照片批量提取内嵌 MP4 视频",
        epilog="示例：uv run python mvimg_to_live.py /Volumes/Share/等待导入照片 --out ~/导出",
    )
    parser.add_argument("src", nargs="?", default="/Volumes/Share/等待导入照片",
                        help="源目录（默认：/Volumes/Share/等待导入照片）")
    parser.add_argument("--out", default=str(Path.home() / "导出"),
                        help="输出目录（默认：~/导出）")
    parser.add_argument("--workers", type=int, default=2, metavar="N",
                        help="并发线程数（默认：2）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出待处理文件，不执行提取")
    args = parser.parse_args()

    if not Path(EXIFTOOL).exists():
        log.error("缺少 exiftool，请运行：brew install exiftool")
        sys.exit(1)

    src_dir = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not src_dir.exists():
        log.error("源目录不存在：%s", src_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("源目录：%s", src_dir)
    log.info("输出目录：%s", out_dir)

    files = find_mvimg_files(src_dir)
    log.info("找到 %d 个 MVIMG 动态照片", len(files))

    if not files:
        log.info("没有找到需要处理的文件。")
        return

    if args.dry_run:
        for f in files:
            print(f)
        return

    success = fail = 0
    if args.workers <= 1:
        for f in files:
            (success if extract_mp4(f, out_dir) else fail)
            success += 1 if extract_mp4(f, out_dir) else 0
    else:
        # 避免上面的 bug，重写
        success = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract_mp4, f, out_dir): f for f in files}
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                    else:
                        fail += 1
                except Exception as e:
                    log.error("处理 %s 时出错：%s", futures[future].name, e)
                    fail += 1

    log.info("━" * 48)
    log.info("完成！成功 %d 个，跳过/失败 %d 个", success, fail)
    log.info("输出目录：%s", out_dir)


if __name__ == "__main__":
    main()
