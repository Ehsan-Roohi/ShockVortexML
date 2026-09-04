"""Local, non-destructive browser annotation tool for Stage-2 review masks.

The server binds to loopback only, reads ``annotations_v0`` as immutable seed
data, and writes reviewed masks plus provenance to ``annotations_v1_working``.
It intentionally keeps shock and vortex masks non-exclusive.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent / "annotation_ui"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "annotations_v0" / "annotation_manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "annotations_v1_working"

EDITABLE_LAYERS = ("shock", "vortex_core", "shock_ignore", "vortex_ignore")
KNOWN_EDITABLE_LAYERS = (
    "shock",
    "shock_centerline",
    "vortex_core",
    "expansion_fan",
    "expansion_mixed",
    "shock_ignore",
    "vortex_ignore",
    "expansion_ignore",
)
VISIBLE_LAYERS = EDITABLE_LAYERS + ("geometry",)
EVIDENCE_CHANNELS = {
    "schlieren": ("diagnostics", 0, "positive"),
    "pressure_gradient": ("diagnostics", 1, "positive"),
    "compression": ("diagnostics", 2, "positive"),
    "vorticity": ("diagnostics", 3, "signed"),
    "q_deviatoric": ("diagnostics", 4, "signed"),
    "lambda_ci": ("diagnostics", 5, "positive"),
}
OUTPUT_SUFFIX = {
    "shock": "shock",
    "shock_centerline": "shock_centerline",
    "vortex_core": "vortex",
    "expansion_fan": "expansion_fan",
    "expansion_mixed": "expansion_mixed",
    "shock_ignore": "shock_ignore",
    "vortex_ignore": "vortex_ignore",
    "expansion_ignore": "expansion_ignore",
    "background_other": "background",
}
REVIEW_STATUSES = {"unreviewed", "in_progress", "reviewed", "needs_followup"}
DECISIONS = {"pending", "accepted", "corrected", "rejected", "not_applicable"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def encode_png(mask: np.ndarray) -> bytes:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def decode_mask_data_url(value: str, expected_shape: tuple[int, int]) -> np.ndarray:
    match = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/=\r\n]+)", value)
    if match is None:
        raise ValueError("mask must be a PNG data URL")
    try:
        raw = base64.b64decode(match.group(1), validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (expected_shape[1], expected_shape[0]):
                raise ValueError(
                    f"mask size {image.size} does not match expected "
                    f"{(expected_shape[1], expected_shape[0])}"
                )
            if image.mode in {"RGBA", "LA"}:
                alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
                return alpha > 127
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            return gray > 127
    except (ValueError, OSError) as exc:
        raise ValueError(f"invalid PNG mask: {exc}") from exc


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def geometry_array(arrays: Any) -> np.ndarray:
    """Return a geometry mask from any supported processed-frame schema."""
    for key in ("geometry_mask", "body_mask", "geometry"):
        if key in arrays:
            return arrays[key]
    raise ValueError("source frame has no geometry_mask, body_mask, or geometry array")


class AnnotationProject:
    def __init__(self, manifest_path: Path, output_dir: Path):
        self.manifest_path = manifest_path.resolve()
        self.output_dir = output_dir.resolve()
        self.seed_dir = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)

        self.editable_layers = tuple(self.manifest.get("editable_layers", EDITABLE_LAYERS))
        unknown_layers = sorted(set(self.editable_layers) - set(KNOWN_EDITABLE_LAYERS))
        if unknown_layers:
            raise ValueError(f"annotation manifest has unknown editable layers: {unknown_layers}")
        if not self.editable_layers:
            raise ValueError("annotation manifest has no editable layers")
        self.visible_layers = self.editable_layers + ("geometry",)
        self.review_heads = tuple(self.manifest.get("review_heads", ("shock", "vortex")))
        if not self.review_heads or not set(self.review_heads) <= {"shock", "vortex", "expansion"}:
            raise ValueError("review_heads must be a non-empty subset of shock/vortex/expansion")

        frames = self.manifest.get("frames", [])
        if not frames:
            raise ValueError("annotation manifest contains no frames")
        self.frames = {int(frame["step"]): frame for frame in frames}
        if len(self.frames) != len(frames):
            raise ValueError("annotation manifest contains duplicate steps")
        self.steps = [int(step) for step in self.manifest.get("selected_steps", self.frames)]
        if set(self.steps) != set(self.frames):
            raise ValueError("selected_steps and frame records disagree")

        self.index = self._load_dataset_index()
        self.diagnostic_stats = self.manifest.get("diagnostic_stats") or self.index.get(
            "diagnostic_normalization_all_frames_for_inspection_only", {}
        )
        self.rows = self._load_review_rows()
        self._evidence_cache: dict[tuple[int, str, float], bytes] = {}

        shape = self.frame_shape(self.steps[0])
        for step in self.steps[1:]:
            if self.frame_shape(step) != shape:
                raise ValueError("review frames do not share a common raster shape")
        self.shape = shape

    def _load_dataset_index(self) -> dict[str, Any]:
        index_value = self.manifest.get("dataset_index")
        if not index_value:
            return {}
        index_path = Path(index_value)
        with index_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _default_row(self, frame: dict[str, Any]) -> dict[str, str]:
        row = {
            "step": str(int(frame["step"])),
            "time": str(float(frame["time"])),
            "review_status": str(frame.get("review_status", "unreviewed")),
            "needs_pixel_correction": "",
            "reviewer": str(frame.get("reviewer", "")),
            "review_notes": str(frame.get("review_notes", "")),
            "updated_utc": "",
        }
        for head in self.review_heads:
            row[f"{head}_decision"] = "pending"
        return row

    def _load_review_rows(self) -> dict[int, dict[str, str]]:
        rows = {step: self._default_row(frame) for step, frame in self.frames.items()}
        candidates = [
            self.output_dir / "review_status.csv",
            self.seed_dir / "review_status.csv",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8-sig", newline="") as stream:
                for incoming in csv.DictReader(stream):
                    step = int(incoming["step"])
                    if step not in rows:
                        continue
                    row = rows[step]
                    for key in row:
                        if key in incoming and incoming[key] is not None:
                            row[key] = incoming[key]
                    # Backward compatibility with annotations_v0 column names.
                    if "shock" in self.review_heads and incoming.get("accept_shock"):
                        row["shock_decision"] = incoming["accept_shock"]
                    if "vortex" in self.review_heads and incoming.get("accept_vortex"):
                        row["vortex_decision"] = incoming["accept_vortex"]
            break
        return rows

    def frame_shape(self, step: int) -> tuple[int, int]:
        source = Path(self.frames[step]["source_frame"])
        with np.load(source, allow_pickle=False) as arrays:
            return tuple(int(value) for value in geometry_array(arrays).shape)

    def load_geometry(self, step: int) -> np.ndarray:
        source = Path(self.frames[step]["source_frame"])
        with np.load(source, allow_pickle=False) as arrays:
            return geometry_array(arrays).astype(bool)

    def output_mask_path(self, step: int, layer: str) -> Path:
        return self.output_dir / "masks" / f"step_{step:05d}_{OUTPUT_SUFFIX[layer]}.png"

    def load_mask(self, step: int, layer: str) -> np.ndarray:
        if layer == "geometry":
            return self.load_geometry(step)
        if layer not in self.editable_layers:
            raise KeyError(layer)
        output = self.output_mask_path(step, layer)
        source = output if output.exists() else Path(self.frames[step]["mask_files"][layer])
        with Image.open(source) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127
        if mask.shape != self.shape:
            raise ValueError(f"mask {source} has shape {mask.shape}, expected {self.shape}")
        return mask

    def mask_png(self, step: int, layer: str) -> bytes:
        return encode_png(self.load_mask(step, layer))

    def render_evidence(self, step: int, name: str, contrast: float) -> bytes:
        if name not in EVIDENCE_CHANNELS:
            raise KeyError(name)
        contrast = float(np.clip(contrast, 0.25, 4.0))
        cache_key = (step, name, round(contrast, 2))
        cached = self._evidence_cache.get(cache_key)
        if cached is not None:
            return cached

        array_group, channel_index, mode = EVIDENCE_CHANNELS[name]
        source = Path(self.frames[step]["source_frame"])
        with np.load(source, allow_pickle=False) as arrays:
            if array_group in arrays:
                values = arrays[array_group][channel_index].astype(np.float32)
            elif "fields" in arrays:
                names = [str(value) for value in arrays["field_names"]]
                fields = arrays["fields"]
                field = {field_name: fields[index].astype(np.float32) for index, field_name in enumerate(names)}
                x = arrays["x"].astype(np.float64)
                y = arrays["y"].astype(np.float64)
                if name == "schlieren":
                    values = field["schlieren"]
                elif name == "pressure_gradient":
                    gy, gx = np.gradient(np.log(np.maximum(field["pressure"], 1e-8)), y, x)
                    values = np.hypot(gx, gy).astype(np.float32)
                elif name == "compression":
                    du_dx = np.gradient(field["u"], x, axis=1)
                    dv_dy = np.gradient(field["v"], y, axis=0)
                    values = np.maximum(-(du_dx + dv_dy), 0.0).astype(np.float32)
                elif name == "vorticity":
                    values = field["omega_z"]
                elif name == "q_deviatoric":
                    values = field["q_criterion"]
                else:
                    values = field["lambda_ci"]
            else:
                raise ValueError(f"source frame {source} has no supported evidence arrays")
            geometry = geometry_array(arrays).astype(bool)
        values = np.nan_to_num(values, copy=False)

        stat_name = {
            0: "grad_log_rho",
            1: "grad_log_pressure",
            2: "compression",
            3: "vorticity",
            4: "q_deviatoric",
            5: "lambda_ci",
        }[channel_index]
        std = float(self.diagnostic_stats.get(stat_name, {}).get("std", 1.0))
        scale = max(std, 1e-6)

        if mode == "positive":
            strength = 1.0 - np.exp(-np.maximum(values, 0.0) * contrast / scale)
            gray = np.clip(255.0 * (1.0 - strength), 0, 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            signed = np.clip(values * contrast / (6.0 * scale), -1.0, 1.0)
            magnitude = np.abs(signed)
            red = np.where(signed >= 0, 255, 255 * (1.0 - magnitude))
            blue = np.where(signed <= 0, 255, 255 * (1.0 - magnitude))
            green = 255 * (1.0 - magnitude)
            rgb = np.stack([red, green, blue], axis=2).astype(np.uint8)

        # The body is shown in neutral gray; it remains a locked mask in the UI.
        rgb[geometry] = (185, 185, 185)
        stream = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(stream, format="PNG", optimize=True)
        result = stream.getvalue()
        self._evidence_cache[cache_key] = result
        return result

    def public_manifest(self) -> dict[str, Any]:
        frames = []
        for step in self.steps:
            frame = self.frames[step]
            row = self.rows[step]
            public_frame = {
                    "step": step,
                    "time": float(frame["time"]),
                    "status": row["review_status"],
                    "needs_pixel_correction": row["needs_pixel_correction"],
                    "reviewer": row["reviewer"],
                    "review_notes": row["review_notes"],
                    "dataset_id": frame.get("dataset_id", f"step_{step:05d}"),
                    "case": frame.get("case", ""),
                    "source_step": frame.get("source_step", step),
                    "reynolds": frame.get("Re_c"),
                    "grid": frame.get("grid", ""),
                    "angle_deg": frame.get("angle_deg"),
                    "source_qualification": frame.get("source_qualification", ""),
                    "quantitative_locus_audit_eligible": bool(
                        frame.get("quantitative_locus_audit_eligible", False)
                    ),
                    "evaluation_role": frame.get("evaluation_role", ""),
                    "selection_roles": frame.get("selection_roles", []),
                    "leakage_group_id": frame.get("leakage_group_id", self.manifest["case_group_id"]),
                    "shock_loss_eligible": bool(frame.get("shock_loss_eligible", True)),
                    "provisional_vortex_track_ids": frame.get(
                        "provisional_vortex_track_ids", []
                    ),
                    "seed_overlap_pixels": int(frame.get("shock_vortex_overlap_pixels", 0)),
                    "saved": all(
                        self.output_mask_path(step, layer).exists()
                        for layer in self.editable_layers
                    ),
                }
            for head in self.review_heads:
                public_frame[f"{head}_decision"] = row[f"{head}_decision"]
            frames.append(public_frame)
        return {
            "schema_version": "2.0",
            "scope": "human review of physics-seeded masks; no automatic ground-truth claim",
            "case_group_id": self.manifest["case_group_id"],
            "overlap_policy": self.manifest["overlap_policy"],
            "width": self.shape[1],
            "height": self.shape[0],
            "layers": list(self.visible_layers),
            "editable_layers": list(self.editable_layers),
            "review_heads": list(self.review_heads),
            "evidence": list(EVIDENCE_CHANNELS),
            "output_dir": str(self.output_dir),
            "frames": frames,
        }

    def _write_review_csv(self) -> None:
        columns = [
            "step",
            "time",
            "review_status",
            *(f"{head}_decision" for head in self.review_heads),
            "needs_pixel_correction",
            "reviewer",
            "review_notes",
            "updated_utc",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for step in self.steps:
            writer.writerow({key: self.rows[step].get(key, "") for key in columns})
        atomic_write(self.output_dir / "review_status.csv", stream.getvalue().encode("utf-8"))

    def _write_working_manifest(self) -> None:
        records = []
        for step in self.steps:
            mask_records = {}
            for layer in (*self.editable_layers, "background_other"):
                path = self.output_mask_path(step, layer)
                if path.exists():
                    data = path.read_bytes()
                    with Image.open(io.BytesIO(data)) as image:
                        count = int(np.count_nonzero(np.asarray(image.convert("L")) > 127))
                    mask_records[layer] = {
                        "path": str(path),
                        "sha256": sha256_bytes(data),
                        "positive_pixels": count,
                    }
            records.append(
                {
                    "step": step,
                    "time": float(self.frames[step]["time"]),
                    "review": self.rows[step],
                    "masks": mask_records,
                }
            )
        payload = {
            "schema_version": "2.0",
            "updated_utc": utc_now(),
            "status": "working human-review set; not frozen ground truth",
            "seed_manifest": str(self.manifest_path),
            "case_group_id": self.manifest["case_group_id"],
            "overlap_policy": self.manifest["overlap_policy"],
            "editable_layers": list(self.editable_layers),
            "review_heads": list(self.review_heads),
            "geometry_policy": "all editable masks are removed inside the exact body mask",
            "frames": records,
        }
        data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        atomic_write(self.output_dir / "annotation_manifest.json", data)

    def save_review(self, step: int, payload: dict[str, Any]) -> dict[str, Any]:
        if step not in self.frames:
            raise KeyError(step)
        status = str(payload.get("review_status", "in_progress"))
        decisions = {
            head: str(payload.get(f"{head}_decision", "pending"))
            for head in self.review_heads
        }
        reviewer = str(payload.get("reviewer", "")).strip()
        notes = str(payload.get("review_notes", "")).strip()
        needs_correction = bool(payload.get("needs_pixel_correction", False))
        if status not in REVIEW_STATUSES:
            raise ValueError(f"invalid review_status: {status}")
        if any(decision not in DECISIONS for decision in decisions.values()):
            raise ValueError("invalid head decision")
        if status == "reviewed":
            if not reviewer:
                raise ValueError("reviewer is required before status can be reviewed")
            if "pending" in decisions.values():
                raise ValueError("all head decisions are required for reviewed status")
            if needs_correction:
                raise ValueError("reviewed status cannot retain needs_pixel_correction")

        incoming = payload.get("masks")
        if not isinstance(incoming, dict) or set(incoming) != set(self.editable_layers):
            raise ValueError(f"masks must contain exactly {list(self.editable_layers)}")

        geometry = self.load_mask(step, "geometry")
        masks = {
            layer: decode_mask_data_url(str(incoming[layer]), self.shape)
            for layer in self.editable_layers
        }
        positive_layers = [layer for layer in self.editable_layers if not layer.endswith("_ignore")]
        removed_inside_geometry = {
            layer: int(np.count_nonzero(masks[layer] & geometry)) for layer in positive_layers
        }
        for layer in self.editable_layers:
            masks[layer] &= ~geometry

        if "shock_centerline" in masks:
            masks["shock"] |= masks["shock_centerline"]
        if "shock_ignore" in masks:
            masks["shock_ignore"] &= ~masks["shock"]
        if "vortex_ignore" in masks:
            masks["vortex_ignore"] &= ~masks["vortex_core"]
        if "expansion_fan" in masks and "expansion_mixed" in masks:
            masks["expansion_mixed"] &= ~masks["expansion_fan"]
        if "expansion_fan" in masks:
            masks["expansion_fan"] &= ~masks["shock"]
        if "expansion_mixed" in masks:
            masks["expansion_mixed"] &= ~masks["shock"]
        if "expansion_ignore" in masks:
            expansion_positive = masks.get("expansion_fan", False) | masks.get(
                "expansion_mixed", False
            )
            masks["expansion_ignore"] &= ~expansion_positive

        background = ~(geometry | np.logical_or.reduce(list(masks.values())))
        written: dict[str, dict[str, Any]] = {}
        for layer, mask in {**masks, "background_other": background}.items():
            data = encode_png(mask)
            path = self.output_mask_path(step, layer)
            atomic_write(path, data)
            written[layer] = {
                "path": str(path),
                "sha256": sha256_bytes(data),
                "positive_pixels": int(np.count_nonzero(mask)),
            }

        row = self.rows[step]
        row.update(
            {
                "review_status": status,
                "needs_pixel_correction": "yes" if needs_correction else "no",
                "reviewer": reviewer,
                "review_notes": notes,
                "updated_utc": utc_now(),
            }
        )
        for head, decision in decisions.items():
            row[f"{head}_decision"] = decision
        self._write_review_csv()
        self._write_working_manifest()
        return {
            "ok": True,
            "step": step,
            "review_status": status,
            "removed_inside_geometry": removed_inside_geometry,
            "shock_vortex_overlap_pixels": int(np.count_nonzero(
                masks.get("shock", False) & masks.get("vortex_core", False)
            )),
            "written": written,
        }

    def check(self) -> dict[str, Any]:
        checked = []
        for step in self.steps:
            masks = {layer: self.load_mask(step, layer) for layer in self.visible_layers}
            evidence = self.render_evidence(step, "schlieren", 1.0)
            checked.append(
                {
                    "step": step,
                    "mask_counts": {key: int(np.count_nonzero(value)) for key, value in masks.items()},
                    "evidence_bytes": len(evidence),
                }
            )
        return {
            "ok": True,
            "frames": len(checked),
            "shape": list(self.shape),
            "case_group_id": self.manifest["case_group_id"],
            "details": checked,
        }


class AnnotationHandler(BaseHTTPRequestHandler):
    project: AnnotationProject
    server_version = "ShockVortexAnnotation/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "frames": len(self.project.steps)})
                return
            if path == "/api/manifest":
                self._send_json(self.project.public_manifest())
                return

            match = re.fullmatch(r"/api/frame/(\d+)/mask/([a-z_]+)\.png", path)
            if match:
                step, layer = int(match.group(1)), match.group(2)
                if step not in self.project.frames or layer not in self.project.visible_layers:
                    self._error(HTTPStatus.NOT_FOUND, "unknown frame or layer")
                    return
                self._send_bytes(self.project.mask_png(step, layer), "image/png")
                return

            match = re.fullmatch(r"/api/frame/(\d+)/evidence/([a-z_]+)\.png", path)
            if match:
                step, evidence = int(match.group(1)), match.group(2)
                if step not in self.project.frames or evidence not in EVIDENCE_CHANNELS:
                    self._error(HTTPStatus.NOT_FOUND, "unknown frame or evidence channel")
                    return
                query = parse_qs(parsed.query)
                contrast = float(query.get("contrast", ["1.0"])[0])
                self._send_bytes(
                    self.project.render_evidence(step, evidence, contrast), "image/png"
                )
                return

            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path in static_files:
                filename, content_type = static_files[path]
                self._send_bytes((UI_DIR / filename).read_bytes(), content_type)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (KeyError, ValueError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/frame/(\d+)/save", path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 32 * 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.project.save_review(int(match.group(1)), payload)
            self._send_json(result)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--check", action="store_true", help="Validate every seed frame and exit without serving"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("annotation server may only bind to a loopback host")
    project = AnnotationProject(args.manifest, args.output_dir)
    if args.check:
        print(json.dumps(project.check(), indent=2))
        return
    AnnotationHandler.project = project
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Annotation UI: http://{args.host}:{args.port}")
    print(f"Seed (read-only): {project.seed_dir}")
    print(f"Working output: {project.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
