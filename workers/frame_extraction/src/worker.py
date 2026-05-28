"""
Frame Extraction Worker

Input:  video artifact (local:// URI)
Output: frames artifact (directory of JPGs) + metadata JSON

Extracts evenly-spaced frames from video using ffmpeg.
Frame count is determined by scene type and video duration.
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path

from sdk.atlas_sdk import BaseWorker, get_logger
from schemas.jobs import Job, JobType

log = get_logger("atlas.worker.frame_extraction")

# Target frame counts per scene type
# Balance: more frames = better reconstruction but slower COLMAP
FRAME_TARGETS: dict[str, int] = {
    "farm_aerial":            150,
    "orchard":                100,
    "vineyard":               100,
    "bridge":                 120,
    "tunnel":                 200,   # tunnels need dense coverage
    "mine":                   180,
    "quarry":                 120,
    "building":               100,
    "facade":                  80,
    "roof":                    80,
    "pipeline":               160,
    "default":                 80,
}


class FrameExtractionWorker(BaseWorker):

    job_type = JobType.FRAME_EXTRACTION

    async def process(self, job: Job, workdir: Path) -> dict[str, str]:
        # ── Download video ────────────────────────────────────────────────────
        video_path = workdir / "input.mp4"
        self.download_input_file(job, "video", video_path)

        # ── Probe video ───────────────────────────────────────────────────────
        probe   = self._probe_video(video_path)
        duration = probe["duration"]
        fps      = probe["fps"]
        width    = probe["width"]
        height   = probe["height"]

        log.info(
            f"Video: {duration:.1f}s, {fps:.1f}fps, "
            f"{width}x{height}"
        )

        # ── Compute extraction fps ─────────────────────────────────────────────
        target = FRAME_TARGETS.get(job.scene_type, FRAME_TARGETS["default"])
        # Override from job metadata if provided
        target = job.metadata.get("target_frames", target)
        extract_fps = min(target / duration, fps)  # never exceed source fps

        log.info(f"Extracting ~{target} frames at {extract_fps:.3f}fps")

        # ── Extract frames ────────────────────────────────────────────────────
        frames_dir = workdir / "frames"
        frames_dir.mkdir()

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf",    f"fps={extract_fps:.6f}",
            "-q:v",   "2",             # high quality JPEG
            "-f",     "image2",
            str(frames_dir / "frame_%05d.jpg"),
            "-y",
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        frames = sorted(frames_dir.glob("*.jpg"))
        if len(frames) < 10:
            raise RuntimeError(
                f"Too few frames extracted: {len(frames)}. "
                f"Check video file integrity."
            )

        log.info(f"Extracted {len(frames)} frames")

        # ── Validate frame quality ────────────────────────────────────────────
        quality = self._assess_quality(frames)
        if quality["blur_ratio"] > 0.4:
            log.warning(
                f"High blur detected in {quality['blur_ratio']:.0%} of frames. "
                f"Reconstruction quality may be reduced."
            )

        # ── Build metadata ────────────────────────────────────────────────────
        metadata = {
            "n_frames":    len(frames),
            "duration_s":  duration,
            "source_fps":  fps,
            "extract_fps": extract_fps,
            "resolution":  [width, height],
            "scene_type":  job.scene_type,
            "quality":     quality,
        }

        # ── Upload outputs ────────────────────────────────────────────────────
        frames_uri = self.upload_output(
            job, frames_dir, "frames", ""
        )
        meta_uri = self.upload_output_json(
            job, metadata, "frames", "metadata.json"
        )

        return {
            "frames":         frames_uri,
            "frames_metadata": meta_uri,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _probe_video(self, path: Path) -> dict:
        """Extract video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"),
            None
        )
        if not video_stream:
            raise RuntimeError("No video stream found in file")

        # Parse fps fraction (e.g. "30/1" or "60000/1001")
        fps_parts = video_stream["r_frame_rate"].split("/")
        fps = int(fps_parts[0]) / int(fps_parts[1])

        return {
            "duration": float(data["format"]["duration"]),
            "fps":      fps,
            "width":    int(video_stream["width"]),
            "height":   int(video_stream["height"]),
            "codec":    video_stream["codec_name"],
        }

    def _assess_quality(self, frames: list[Path]) -> dict:
        """
        Quick quality assessment on a sample of frames.
        Detects blur using Laplacian variance.
        Low variance = blurry frame.
        """
        try:
            import cv2
            import numpy as np

            # Sample every 5th frame
            sample = frames[::5][:20]
            blur_threshold = 100.0  # tune per scene type
            blurry = 0

            for frame_path in sample:
                img  = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                var  = cv2.Laplacian(img, cv2.CV_64F).var()
                if var < blur_threshold:
                    blurry += 1

            blur_ratio = blurry / len(sample) if sample else 0.0
            return {
                "blur_ratio":       round(blur_ratio, 3),
                "blur_threshold":   blur_threshold,
                "frames_sampled":   len(sample),
            }
        except ImportError:
            log.warning("cv2 not available — skipping quality assessment")
            return {"blur_ratio": 0.0, "frames_sampled": 0}


