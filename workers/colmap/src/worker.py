"""
COLMAP Worker

Input:  frames artifact
Output: colmap_sparse artifact (cameras.bin, images.bin, points3D.bin)
        colmap_metadata JSON (registration rate, reprojection error, n_points)

Runs Structure from Motion to recover camera poses from frames.
Uses hloc (SuperPoint + SuperGlue) when available, falls back to SIFT.
"""

from __future__ import annotations
import json
import os
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

from atlas_sdk import BaseWorker, get_logger
from jobs import Job, JobType

log = get_logger("atlas.worker.colmap")

# Matching strategy per scene type
# sequential: video/drone flyover — frames are temporally ordered
# exhaustive: turntable/orbit — every frame vs every frame
MATCHING_STRATEGY: dict[str, str] = {
    "farm_aerial":  "sequential",
    "orchard":      "sequential",
    "vineyard":     "sequential",
    "bridge":       "exhaustive",
    "tunnel":       "sequential",
    "mine":         "sequential",
    "quarry":       "sequential",
    "building":     "exhaustive",
    "facade":       "exhaustive",
    "roof":         "sequential",
    "pipeline":     "sequential",
    "default":      "sequential",
}

# Minimum acceptable registration rate per scene type
MIN_REGISTRATION_RATE: dict[str, float] = {
    "tunnel":  0.70,   # tunnels are hard — accept lower
    "mine":    0.70,
    "default": 0.80,
}


class ColmapWorker(BaseWorker):

    job_type = JobType.COLMAP

    def __init__(self):
        super().__init__()
        self._use_hloc = False

    async def process(self, job: Job, workdir: Path) -> dict[str, str]:
        # ── Download frames ───────────────────────────────────────────────────
        frames_dir = self.download_input(job, "frames", workdir)
        frames = sorted(frames_dir.glob("*.jpg"))
        log.info(f"Running COLMAP on {len(frames)} frames")

        # ── Setup COLMAP paths ────────────────────────────────────────────────
        colmap_dir  = workdir / "colmap"
        db_path     = colmap_dir / "database.db"
        sparse_dir  = colmap_dir / "sparse"
        colmap_dir.mkdir()
        sparse_dir.mkdir()

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("DISPLAY", "")

        strategy = MATCHING_STRATEGY.get(
            job.scene_type,
            MATCHING_STRATEGY["default"]
        )
        # Allow job-level override
        strategy = job.metadata.get("matching_strategy", strategy)

        log.info(f"Matching strategy: {strategy}")

        # ── Run reconstruction ────────────────────────────────────────────────
        if self._use_hloc:
            log.info("Using hloc (SuperPoint + SuperGlue)")
            self._run_hloc(frames_dir, colmap_dir, strategy)
        else:
            log.info("Using COLMAP SIFT")
            self._run_colmap_sift(frames_dir, db_path, sparse_dir, strategy)

        # ── Validate output ───────────────────────────────────────────────────
        sparse_models = sorted(sparse_dir.iterdir())
        if not sparse_models:
            raise RuntimeError(
                "COLMAP produced no reconstruction. "
                "Possible causes: insufficient frame overlap, "
                "too-fast camera movement, featureless scene. "
                "Try exhaustive matching or re-record footage."
            )

        # Use largest model (most registered images)
        best_model = max(
            sparse_models,
            key=lambda p: self._count_registered(p / "images.bin")
        )

        n_registered = self._count_registered(best_model / "images.bin")
        n_total      = len(frames)
        reg_rate     = n_registered / n_total

        min_rate = job.metadata.get(
            "min_registration_rate",
            MIN_REGISTRATION_RATE.get(
                job.scene_type,
                MIN_REGISTRATION_RATE["default"],
            ),
        )
        if reg_rate < min_rate:
            raise RuntimeError(
                f"Poor registration: {n_registered}/{n_total} frames "
                f"({reg_rate:.0%} < {min_rate:.0%} minimum). "
                f"Try exhaustive matching or improve footage quality."
            )

        log.info(f"Registered {n_registered}/{n_total} frames ({reg_rate:.0%})")

        # ── Extract metadata ──────────────────────────────────────────────────
        n_points    = self._count_points(best_model / "points3D.bin")
        repr_error  = self._mean_reprojection_error(best_model / "points3D.bin")
        db_stats    = self._db_stats(db_path) if db_path.exists() else {}

        metadata = {
            "n_registered":       n_registered,
            "n_total_frames":     n_total,
            "registration_rate":  round(reg_rate, 4),
            "n_sparse_points":    n_points,
            "mean_reproj_error":  round(repr_error, 4),
            "matching_strategy":  strategy,
            "used_hloc":          self._use_hloc,
            "db_stats":           db_stats,
        }

        # Quality warning thresholds
        if n_points < 1000:
            log.warning(f"Low point count: {n_points}. Scene may reconstruct poorly.")
        if repr_error > 2.0:
            log.warning(f"High reprojection error: {repr_error:.2f}px")

        # ── Upload outputs ────────────────────────────────────────────────────
        sparse_uri = self.upload_output(
            job, best_model, "colmap_sparse", ""
        )
        meta_uri = self.upload_output_json(
            job, metadata, "colmap_sparse", "metadata.json"
        )

        return {
            "colmap_sparse":          sparse_uri,
            "colmap_sparse_metadata": meta_uri,
        }

    # ── COLMAP SIFT pipeline ──────────────────────────────────────────────────

    def _colmap_threads(self) -> str:
        # COLMAP 4.x on macOS can SIGSEGV with multi-threaded SIFT matching.
        if os.uname().sysname == "Darwin":
            return "1"
        return "4"

    def _run_colmap_sift(
        self,
        images_dir: Path,
        db_path:    Path,
        sparse_dir: Path,
        strategy:   str,
    ):
        env = {"QT_QPA_PLATFORM": "offscreen", **os.environ}
        n_threads = self._colmap_threads()

        log.info(f"COLMAP using {n_threads} thread(s) for SIFT")

        # Feature extraction (COLMAP 4.x option names)
        log.info("COLMAP: feature_extractor")
        self._run(["colmap", "feature_extractor",
            "--database_path",              str(db_path),
            "--image_path",                 str(images_dir),
            "--ImageReader.single_camera",  "1",
            "--FeatureExtraction.use_gpu",  "0",
            "--FeatureExtraction.num_threads", n_threads,
        ], env=env)

        self._run_colmap_matching(db_path, strategy, env, n_threads)

        # Reconstruction
        log.info("COLMAP: mapper")
        self._run(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path",    str(images_dir),
            "--output_path",   str(sparse_dir),
        ], env=env)

    def _run_colmap_matching(
        self,
        db_path: Path,
        strategy: str,
        env: dict,
        n_threads: str,
    ):
        match_base = [
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.num_threads", n_threads,
        ]
        # Homebrew COLMAP 4.x on macOS SIGSEGVs in the default SIFT matcher path.
        if os.uname().sysname == "Darwin":
            match_base.extend([
                "--SiftMatching.cpu_brute_force_matcher", "1",
            ])
        strategies = (
            [strategy]
            if strategy == "exhaustive"
            else [strategy, "exhaustive"]
        )
        last_err: RuntimeError | None = None

        for attempt in strategies:
            try:
                if attempt == "exhaustive":
                    log.info("COLMAP: exhaustive_matcher")
                    self._run(
                        ["colmap", "exhaustive_matcher", *match_base],
                        env=env,
                    )
                else:
                    log.info("COLMAP: sequential_matcher")
                    self._run([
                        "colmap", "sequential_matcher",
                        *match_base,
                        "--SequentialMatching.overlap", "15",
                        "--SequentialMatching.loop_detection", "0",
                    ], env=env)
                return
            except RuntimeError as e:
                last_err = e
                log.warning(f"COLMAP {attempt} matching failed: {e}")

        if last_err:
            raise last_err

    # ── hloc pipeline (better quality, optional) ──────────────────────────────

    def _run_hloc(self, images_dir: Path, colmap_dir: Path, strategy: str):
        """
        Use hloc for feature extraction and matching.
        SuperPoint keypoints + SuperGlue matcher.
        Significantly better on repetitive textures (crop rows, brick, etc.)
        Falls back to COLMAP mapper for triangulation.
        """
        try:
            from hloc import extract_features, match_features, reconstruction
            from hloc.utils.parsers import names_to_pair

            features_path = colmap_dir / "features.h5"
            matches_path  = colmap_dir / "matches.h5"
            pairs_path    = colmap_dir / "pairs.txt"
            sparse_dir    = colmap_dir / "sparse"

            # Generate pairs
            if strategy == "exhaustive":
                from hloc.pairs_from_exhaustive import main as pairs_exhaustive
                pairs_exhaustive(pairs_path, image_list=images_dir)
            else:
                from hloc.pairs_from_sequential import main as pairs_sequential
                pairs_sequential(pairs_path, image_list=images_dir, window_size=15)

            # Extract features
            extract_features.main(
                conf        = extract_features.confs["superpoint_aachen"],
                image_dir   = images_dir,
                feature_path = features_path,
            )

            # Match features
            match_features.main(
                conf          = match_features.confs["superglue"],
                pairs         = pairs_path,
                features      = features_path,
                matches       = matches_path,
            )

            # Reconstruct using COLMAP mapper
            reconstruction.main(
                sfm_dir     = sparse_dir,
                image_dir   = images_dir,
                pairs       = pairs_path,
                features    = features_path,
                matches     = matches_path,
            )

        except Exception as e:
            log.warning(f"hloc failed: {e}. Falling back to COLMAP SIFT.")
            db_path = colmap_dir / "database.db"
            sparse_dir = colmap_dir / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            self._run_colmap_sift(images_dir, db_path, sparse_dir, strategy)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], env: dict | None = None):
        result = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
            env            = env or os.environ,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd[:3])}\n"
                f"stderr: {result.stderr[-2000:]}"  # last 2000 chars
            )

    def _count_registered(self, images_bin: Path) -> int:
        if not images_bin.exists():
            return 0
        with open(images_bin, "rb") as f:
            return struct.unpack("Q", f.read(8))[0]

    def _count_points(self, points3d_bin: Path) -> int:
        if not points3d_bin.exists():
            return 0
        with open(points3d_bin, "rb") as f:
            return struct.unpack("Q", f.read(8))[0]

    def _mean_reprojection_error(self, points3d_bin: Path) -> float:
        if not points3d_bin.exists():
            return 0.0
        errors = []
        with open(points3d_bin, "rb") as f:
            n = struct.unpack("Q", f.read(8))[0]
            for _ in range(n):
                f.read(8)    # point_id
                f.read(24)   # xyz
                f.read(3)    # rgb
                error = struct.unpack("d", f.read(8))[0]
                errors.append(error)
                track_len = struct.unpack("Q", f.read(8))[0]
                f.read(8 * track_len)
        return float(sum(errors) / len(errors)) if errors else 0.0

    def _db_stats(self, db_path: Path) -> dict:
        try:
            conn = sqlite3.connect(str(db_path))
            stats = {
                "n_images":   conn.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                "n_keypoints": conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0],
                "n_matches":  conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            }
            conn.close()
            return stats
        except Exception:
            return {}

    def _check_hloc(self) -> bool:
        try:
            import hloc
            return True
        except ImportError:
            log.info("hloc not installed — using COLMAP SIFT")
            return False


