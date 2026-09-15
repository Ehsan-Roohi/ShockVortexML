"""Render high-fidelity 4K density-schlieren movies from completed MFC restarts.

The raw CFD files are read only.  Every stored continuation state is encoded
once, in time order, with one fixed visualization scale per case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("MFC_SHOWCASE_ROOT", Path.cwd())).resolve()
FPS = 30
SIZE = (3840, 2160)
CASES = {
    "airfoil_m3_a40_re1e4_f270": dict(nx=2970, ny=2700, dt=1/5400, title="Diamond airfoil | M=3 | alpha=40 deg | Re=10,000"),
    "cylinder_m2p7_f180": dict(nx=1980, ny=1800, dt=8/26640, title="Circular cylinder | M=2.7 | Euler"),
    "ellipse_m2p7_f90": dict(nx=990, ny=900, dt=8/13360, title="Elliptical cylinder | M=2.7 | Euler"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def steps_in(directory: Path) -> list[int]:
    rows = []
    for path in directory.glob("lustre_*.dat"):
        match = re.fullmatch(r"lustre_(\d+)\.dat", path.name)
        if match:
            rows.append(int(match.group(1)))
    return sorted(rows)


def font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def sampled_rho(path: Path, cfg: dict, stride: int) -> np.ndarray:
    expected = 5 * cfg["nx"] * cfg["ny"] * 8
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name}: {path.stat().st_size} != {expected}")
    q = np.memmap(path, dtype="<f8", mode="r", shape=(5, cfg["ny"], cfg["nx"]))
    rho = np.asarray(q[0, ::stride, ::stride], dtype=np.float32)
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError(f"invalid density in {path.name}")
    return rho


def gradient(rho: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(rho)
    return np.hypot(gx, gy)


def frame_image(rho: np.ndarray, scale: float, title: str, time_value: float, index: int, total: int) -> Image.Image:
    g = gradient(rho)
    intensity = np.clip(np.log1p(40.0 * g / scale) / np.log1p(40.0), 0.0, 1.0)
    # Traditional light-background numerical schlieren; strong gradients are dark.
    gray = np.asarray(np.rint(255.0 * (1.0 - intensity)), dtype=np.uint8)
    plot = Image.fromarray(gray, mode="L").convert("RGB")
    plot.thumbnail((SIZE[0] - 96, SIZE[1] - 260), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", SIZE, "white")
    x0 = (SIZE[0] - plot.width) // 2
    y0 = 120 + (SIZE[1] - 260 - plot.height) // 2
    canvas.paste(plot, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 28), title, fill="black", font=font(58))
    label = f"density schlieren   t={time_value:.4f}   stored state {index + 1}/{total}"
    draw.text((48, SIZE[1] - 88), label, fill=(30, 30, 30), font=font(46))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    cfg = CASES[args.case]
    case = ROOT / args.case
    raw = case / "production" / "restart_data"
    complete = json.loads((case / "COMPLETE.json").read_text())
    steps = steps_in(raw)
    if len(steps) != complete["states"] or len(steps) < 2 or np.any(np.diff(steps) <= 0):
        raise ValueError("state inventory mismatch")
    # Preserve every source grid point.  The earlier review render sampled the
    # field spatially; the 4K master intentionally performs no such reduction.
    stride = 1
    calibration_indices = np.linspace(0, len(steps) - 1,  nine := 9, dtype=int)
    levels = []
    for i in calibration_indices:
        levels.append(float(np.percentile(gradient(sampled_rho(raw / f"lustre_{steps[i]}.dat", cfg, stride)), 99.7)))
    scale = float(np.median(levels))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid fixed schlieren scale")

    output_dir = case / "videos_hq"
    output_dir.mkdir(exist_ok=True)
    final = output_dir / f"{args.case}_t{steps[0]*cfg['dt']:.0f}_t{steps[-1]*cfg['dt']:.0f}_schlieren.mp4"
    temporary = final.with_suffix(".partial.mp4")
    if final.exists() or temporary.exists():
        raise FileExistsError(final)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv444p", "-movflags", "+faststart", str(temporary)]
    started = time.time()
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview_indices = {0, len(steps)//2, len(steps)-1}
    try:
        for i, step in enumerate(steps):
            image = frame_image(sampled_rho(raw / f"lustre_{step}.dat", cfg, stride), scale, cfg["title"], step*cfg["dt"], i, len(steps))
            if i in preview_indices:
                image.save(output_dir / f"preview_{i:04d}.png")
            process.stdin.write(image.tobytes())
            if (i + 1) % 50 == 0:
                print(args.case, i + 1, "/", len(steps), flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    except BaseException:
        process.kill(); process.wait()
        raise
    temporary.replace(final)
    check = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(final), "-f", "null", "-"], capture_output=True)
    if check.returncode:
        raise RuntimeError(check.stderr.decode("utf8", "replace"))
    receipt = dict(case=args.case, video=final.name, sha256=sha256(final), bytes=final.stat().st_size,
                   states=len(steps), fps=FPS, duration_seconds=len(steps)/FPS, resolution=list(SIZE),
                   source_time=[steps[0]*cfg["dt"], steps[-1]*cfg["dt"]], fixed_scale=scale,
                   scale_calibration_indices=calibration_indices.tolist(), spatial_stride=stride,
                   full_decode_pass=True, raw_files_modified=False, elapsed_seconds=time.time()-started)
    (output_dir / "VIDEO.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
