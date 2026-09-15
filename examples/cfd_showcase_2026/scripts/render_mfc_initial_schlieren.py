"""Render the pre-continuation t=0 segments for the three MFC movies.

Raw CFD files remain read-only. These outputs are joined to the already-rendered
continuations only after the duplicate restart state is removed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from render_schlieren_movies import CASES, FPS, SIZE, font, gradient, sampled_rho, sha256, steps_in
import imageio_ffmpeg
import numpy as np
from PIL import Image
import subprocess
import time

ROOT = Path(os.environ.get("MFC_SHOWCASE_ROOT", Path.cwd())).resolve()
SOURCES = {
    "airfoil_m3_a40_re1e4_f270": Path(os.environ.get("MFC_AIRFOIL_SOURCE", ROOT / "airfoil_m3_a40_re1e4_f270")),
    "cylinder_m2p7_f180": Path(os.environ.get("MFC_CYLINDER_SOURCE", ROOT / "cylinder_m2p7_f180")),
    "ellipse_m2p7_f90": Path(os.environ.get("MFC_ELLIPSE_SOURCE", ROOT / "ellipse_m2p7_f90")),
}
EXPECTED = {"airfoil_m3_a40_re1e4_f270": 121, "cylinder_m2p7_f180": 81, "ellipse_m2p7_f90": 81}


def frame_image(rho, scale):
    g = gradient(rho)
    intensity = np.clip(np.log1p(40.0 * g / scale) / np.log1p(40.0), 0.0, 1.0)
    gray = np.asarray(np.rint(255.0 * (1.0 - intensity)), dtype=np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    cfg = CASES[args.case]
    raw = SOURCES[args.case] / "restart_data"
    steps = steps_in(raw)
    if len(steps) != EXPECTED[args.case] or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError(f"initial state inventory mismatch: {len(steps)}, {steps[:1]}, {steps[-1:]}")
    calibration_indices = np.linspace(0, len(steps) - 1, 9, dtype=int)
    levels = [float(np.percentile(gradient(sampled_rho(raw / f"lustre_{steps[i]}.dat", cfg, 1)), 99.7)) for i in calibration_indices]
    scale = float(np.median(levels))
    output_dir = ROOT / args.case / "videos_initial_hq"
    output_dir.mkdir(exist_ok=True)
    final = output_dir / f"{args.case}_t0_t{steps[-1]*cfg['dt']:.0f}_schlieren.mp4"
    temporary = final.with_suffix(".partial.mp4")
    final.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv444p", "-movflags", "+faststart", str(temporary)]
    started = time.time(); process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i, step in enumerate(steps):
            image = frame_image(sampled_rho(raw / f"lustre_{step}.dat", cfg, 1), scale)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait() != 0: raise RuntimeError("ffmpeg encoding failed")
    except BaseException:
        process.kill(); process.wait(); raise
    temporary.replace(final)
    check = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(final), "-f", "null", "-"], capture_output=True)
    if check.returncode: raise RuntimeError(check.stderr.decode("utf8", "replace"))
    receipt = {"case": args.case, "video": final.name, "sha256": sha256(final), "bytes": final.stat().st_size, "states": len(steps), "fps": FPS, "resolution": list(SIZE), "source_time": [0.0, steps[-1]*cfg["dt"]], "fixed_scale": scale, "spatial_stride": 1, "full_decode_pass": True, "raw_files_modified": False, "elapsed_seconds": time.time()-started}
    (output_dir / "VIDEO.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__": main()
