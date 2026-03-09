#!/usr/bin/env python3
"""
dedup_images.py
~~~~~~~~~~~~~~~
图片去重工具：比对「参考文件夹」与「待匹配文件夹」，将以下图片复制到导出目录：
  1. 待匹配文件夹中**不存在于参考文件夹**的图片（全新图片）
  2. 两边都有同一张图，但**待匹配版本文件更大**（更高质量版本）

匹配策略（两层，逐层降级）：
  Level 1 — EXIF 元数据匹配（快速，默认）
    比对：DateTimeOriginal + ImageWidth + ImageHeight + Make + Model
    优点：速度极快，无需解码图片像素
    缺点：APP 裁剪/重保存后 EXIF 会丢失，平台处理后元数据可能不一致

  Level 2 — 感知哈希（Perceptual Hash，--advanced）
    算法：pHash（DCT 感知哈希），汉明距离阈值可调
    对 JPEG 重压缩、有损编辑、轻微裁剪有良好鲁棒性
    依赖：Pillow、imagehash

用法：
    uv run python img_tools/dedup_images.py <参考目录> <待匹配目录> [--out 导出目录]
                                   [--advanced] [--hash-threshold N]
                                   [--workers N] [--dry-run] [--verbose]

示例：
    # 仅 EXIF 匹配（快速）
    uv run python img_tools/dedup_images.py ~/Photos/Reference ~/Photos/Pending --out ~/Exports/Unique

    # 启用感知哈希（更准确）
    uv run python img_tools/dedup_images.py ~/Photos/Reference ~/Photos/Pending \\
        --out ~/Exports/Unique --advanced --hash-threshold 8
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Optional

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

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tif", ".tiff", ".webp", ".avif", ".bmp",
    ".gif", ".cr2", ".cr3", ".nef", ".arw",
    ".dng", ".raf", ".orf", ".rw2", ".srw",
}

app = typer.Typer(
    name="dedup-images",
    help="图片去重工具：找出待匹配文件夹中不重复或更高质量的图片",
    no_args_is_help=True,
    add_completion=False,
)


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class ImageMeta:
    path: Path
    size: int
    datetime: str | None = None
    width: int | None = None
    height: int | None = None
    make: str | None = None
    model: str | None = None
    phash: str | None = None

    @property
    def meta_key(self) -> tuple | None:
        if self.datetime and self.width and self.height:
            return (self.datetime, self.width, self.height,
                    self.make or "", self.model or "")
        return None


@dataclass
class MatchResult:
    unique: list[ImageMeta] = field(default_factory=list)
    larger_dup: list[ImageMeta] = field(default_factory=list)
    exact_dup: list[ImageMeta] = field(default_factory=list)


# ─── EXIF 读取 ────────────────────────────────────────────────────────────────

_EXIF_TAGS = [
    "-DateTimeOriginal", "-CreateDate",
    "-ImageWidth", "-ImageHeight",
    "-Make", "-Model",
]


def read_exif_batch(paths: list[Path], workers: int = 4) -> dict[Path, ImageMeta]:
    """批量读取 EXIF，返回 {路径: ImageMeta}。"""
    result: dict[Path, ImageMeta] = {}
    batch_size = 500

    for i in range(0, len(paths), batch_size):
        batch = paths[i: i + batch_size]
        cmd = [EXIFTOOL, "-j", "-q"] + _EXIF_TAGS + [str(p) for p in batch]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode not in (0, 1):
            log.warning("exiftool 返回错误", batch=i // batch_size,
                        stderr=r.stderr[:200])
            continue
        try:
            entries = json.loads(r.stdout or "[]")
        except json.JSONDecodeError:
            log.warning("exiftool JSON 解析失败", batch=i // batch_size)
            continue

        for entry in entries:
            src = Path(entry.get("SourceFile", ""))
            if not src.exists():
                continue
            dt = entry.get("DateTimeOriginal") or entry.get("CreateDate")
            w = entry.get("ImageWidth")
            h = entry.get("ImageHeight")
            result[src] = ImageMeta(
                path=src,
                size=src.stat().st_size,
                datetime=str(dt).strip() if dt else None,
                width=int(w) if w else None,
                height=int(h) if h else None,
                make=str(entry.get("Make", "")).strip() or None,
                model=str(entry.get("Model", "")).strip() or None,
            )

    for p in paths:
        if p not in result:
            try:
                result[p] = ImageMeta(path=p, size=p.stat().st_size)
            except OSError:
                pass

    return result


# ─── 感知哈希 ─────────────────────────────────────────────────────────────────

def _compute_phash(img_meta: ImageMeta) -> str | None:
    try:
        import imagehash
        from PIL import Image
        with Image.open(img_meta.path) as im:
            return str(imagehash.phash(im))
    except Exception as e:
        log.debug("pHash 计算失败", file=img_meta.path.name, error=str(e))
        return None


def compute_phash_batch(metas: list[ImageMeta], workers: int = 4) -> None:
    """并发计算 pHash，结果就地写入 meta.phash。"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute_phash, m): m for m in metas}
        for fut in as_completed(futures):
            futures[fut].phash = fut.result()


def phash_distance(a: str, b: str) -> int:
    try:
        import imagehash
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    except Exception:
        return 999


# ─── 文件扫描 ─────────────────────────────────────────────────────────────────

def scan_images(directory: Path) -> list[Path]:
    return [
        f for f in sorted(directory.rglob("*"))
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]


# ─── 核心比对逻辑 ─────────────────────────────────────────────────────────────

def build_meta_index(metas: dict[Path, ImageMeta]) -> dict[tuple, list[ImageMeta]]:
    index: dict[tuple, list[ImageMeta]] = {}
    for m in metas.values():
        key = m.meta_key
        if key is not None:
            index.setdefault(key, []).append(m)
    return index


def find_best_ref_match_by_meta(
    candidate: ImageMeta,
    ref_index: dict[tuple, list[ImageMeta]],
) -> ImageMeta | None:
    key = candidate.meta_key
    if key is None:
        return None
    matches = ref_index.get(key, [])
    if not matches:
        return None
    return min(matches, key=lambda m: abs(m.size - candidate.size))


def compare(
    ref_metas: dict[Path, ImageMeta],
    cand_metas: dict[Path, ImageMeta],
    advanced: bool = False,
    hash_threshold: int = 10,
    workers: int = 4,
) -> MatchResult:
    result = MatchResult()
    ref_index = build_meta_index(ref_metas)

    exif_matched: list[tuple[ImageMeta, ImageMeta]] = []
    exif_no_match: list[ImageMeta] = []

    for cand in cand_metas.values():
        ref_match = find_best_ref_match_by_meta(cand, ref_index)
        if ref_match is not None:
            exif_matched.append((cand, ref_match))
        else:
            exif_no_match.append(cand)

    for cand, ref in exif_matched:
        if cand.size > ref.size:
            result.larger_dup.append(cand)
            log.debug("更大重复", file=cand.path.name,
                      cand_bytes=cand.size, ref_bytes=ref.size)
        else:
            result.exact_dup.append(cand)
            log.debug("跳过重复", file=cand.path.name)

    if not advanced or not exif_no_match:
        result.unique.extend(exif_no_match)
    else:
        log.info("EXIF 未命中，启用 pHash 比对", count=len(exif_no_match))
        log.info("  计算候选图片 pHash…")
        compute_phash_batch(exif_no_match, workers=workers)

        ref_no_exif = [m for m in ref_metas.values() if m.meta_key is None]
        if ref_no_exif:
            log.info("  计算参考图片 pHash", count=len(ref_no_exif))
            compute_phash_batch(ref_no_exif, workers=workers)

        ref_with_hash = [m for m in ref_metas.values() if m.phash]
        if not ref_with_hash:
            log.info("  参考图片无可用 pHash，候选全部视为新图片")
            result.unique.extend(exif_no_match)
        else:
            for cand in exif_no_match:
                if cand.phash is None:
                    result.unique.append(cand)
                    continue
                best_ref: ImageMeta | None = None
                best_dist = hash_threshold + 1
                for ref in ref_with_hash:
                    d = phash_distance(cand.phash, ref.phash)
                    if d <= hash_threshold and d < best_dist:
                        best_dist = d
                        best_ref = ref

                if best_ref is None:
                    log.debug("新图片（pHash 无匹配）", file=cand.path.name)
                    result.unique.append(cand)
                elif cand.size > best_ref.size:
                    log.debug("更大重复（pHash）", file=cand.path.name, dist=best_dist)
                    result.larger_dup.append(cand)
                else:
                    log.debug("跳过重复（pHash）", file=cand.path.name, dist=best_dist)
                    result.exact_dup.append(cand)

    return result


# ─── 导出 ─────────────────────────────────────────────────────────────────────

def export_files(
    items: list[ImageMeta],
    kind: Literal["unique", "larger_dup"],
    out_dir: Path,
    dry_run: bool = False,
) -> int:
    label = {"unique": "新图片", "larger_dup": "更大重复"}[kind]
    dest_base = out_dir / kind
    if not dry_run:
        dest_base.mkdir(parents=True, exist_ok=True)

    count = 0
    for meta in items:
        dest = dest_base / meta.path.name
        if dest.exists():
            stem, suffix = meta.path.stem, meta.path.suffix
            idx = 1
            while dest.exists():
                dest = dest_base / f"{stem}_{idx}{suffix}"
                idx += 1

        if dry_run:
            log.info("[DRY-RUN]", kind=kind, file=meta.path.name, dest=dest.name)
        else:
            shutil.copy2(str(meta.path), str(dest))
            log.info("导出", label=label, src=meta.path.name, dst=dest.name)
        count += 1

    return count


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    ref: Annotated[
        Path,
        typer.Argument(help="参考文件夹（已有照片库）"),
    ],
    target: Annotated[
        Path,
        typer.Argument(help="待匹配文件夹（待整理的图片）"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="导出目录"),
    ] = Path.home() / "导出",
    advanced: Annotated[
        bool,
        typer.Option("--advanced", help="启用感知哈希（pHash）对 EXIF 未命中的图片做二次比对"),
    ] = False,
    hash_threshold: Annotated[
        int,
        typer.Option("--hash-threshold", min=1, max=64,
                     help="pHash 汉明距离阈值（默认 10，越小越严格，建议范围 5–15）"),
    ] = 10,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="并发线程数"),
    ] = 4,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="仅打印将导出的文件，不执行复制"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="输出调试日志"),
    ] = False,
) -> None:
    """
    图片去重：找出待匹配文件夹中不存在于参考库、或质量更高的图片，复制到导出目录。

    \b
    匹配策略：
      Level 1（默认）  EXIF 元数据匹配
        比对字段：拍摄时间 + 分辨率 + 相机品牌/型号
        速度快，EXIF 缺失（截图、APP 处理图等）将被视为新图片。

      Level 2（--advanced）  感知哈希 pHash
        对 EXIF 未命中的图片，通过视觉相似度二次比对。
        可识别 JPEG 重压缩、轻度裁剪等处理后的重复图。
    """
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10)
        )

    if not Path(EXIFTOOL).exists():
        log.error("缺少 exiftool，请运行：brew install exiftool")
        raise typer.Exit(1)

    if advanced:
        try:
            import imagehash  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            log.error("启用 --advanced 需要安装依赖：uv add Pillow imagehash")
            raise typer.Exit(1)

    ref_dir    = ref.expanduser().resolve()
    target_dir = target.expanduser().resolve()
    out_dir    = out.expanduser().resolve()

    for name, p in [("参考目录", ref_dir), ("待匹配目录", target_dir)]:
        if not p.is_dir():
            log.error("目录不存在或不是目录", name=name, path=str(p))
            raise typer.Exit(1)

    # ── 扫描 ──
    log.info("扫描参考目录", path=str(ref_dir))
    ref_paths = scan_images(ref_dir)
    log.info("扫描完成", count=len(ref_paths))

    log.info("扫描待匹配目录", path=str(target_dir))
    target_paths = scan_images(target_dir)
    log.info("扫描完成", count=len(target_paths))

    if not target_paths:
        log.info("待匹配目录中没有图片，退出")
        return

    # ── 读取 EXIF ──
    log.info("读取参考图片 EXIF…")
    ref_metas = read_exif_batch(ref_paths, workers=workers)
    log.info("读取待匹配图片 EXIF…")
    target_metas = read_exif_batch(target_paths, workers=workers)

    # ── 比对 ──
    log.info("开始比对…")
    result = compare(
        ref_metas=ref_metas,
        cand_metas=target_metas,
        advanced=advanced,
        hash_threshold=hash_threshold,
        workers=workers,
    )

    # ── 汇总 ──
    log.info("比对结果",
             unique=len(result.unique),
             larger_dup=len(result.larger_dup),
             skipped=len(result.exact_dup))

    to_export = result.unique + result.larger_dup
    if not to_export:
        log.info("没有需要导出的图片")
        return

    log.info("准备导出", total=len(to_export), out=str(out_dir))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_unique = export_files(result.unique, "unique", out_dir, dry_run=dry_run)
    n_dup    = export_files(result.larger_dup, "larger_dup", out_dir, dry_run=dry_run)

    if dry_run:
        log.info("[DRY-RUN] 将导出", unique=n_unique, larger_dup=n_dup)
    else:
        log.info("完成！",
                 unique=f"{n_unique} → unique/",
                 larger_dup=f"{n_dup} → larger_dup/",
                 out=str(out_dir))


if __name__ == "__main__":
    app()
