"""
Splat Worker (Phase 1)

Input:  frames artifact, colmap_sparse artifact
Output: splat artifact (.ply) + splat_metadata JSON

Trains nerfstudio splatfacto from COLMAP sparse reconstruction.
"""

from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

from sdk.atlas_sdk import BaseWorker, get_logger
from jobs import Job, JobType
from intelligence.src.domains.registry import get_plugin

log = get_logger("atlas.worker.splat")

# Training hyperparameter profiles per scene type
# Override via job.metadata["hyperparameters"]
DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "farm_aerial": {
        "max_num_iterations":      30000,
        "densify_grad_threshold":  0.0002,
        "num_downscales":          2,
        "cull_alpha_thresh":       0.005,
        "use_scale_regularization": True,
    },
    "tunnel": {
        "max_num_iterations":      25000,
        "densify_grad_threshold":  0.0003,
        "num_downscales":          1,
        "cull_alpha_thresh":       0.005,
        "use_scale_regularization": True,
    },
    "mine": {
        "max_num_iterations":      25000,
        "densify_grad_threshold":  0.0003,
        "num_downscales":          1,
        "cull_alpha_thresh":       0.005,
        "use_scale_regularization": True,
    },
    "bridge": {
        "max_num_iterations":      20000,
        "densify_grad_threshold":  0.0004,
        "num_downscales":          0,
        "cull_alpha_thresh":       0.01,
        "use_scale_regularization": True,
    },
    "default": {
        "max_num_iterations":      20000,
        "densify_grad_threshold":  0.0005,
        "num_downscales":          1,
        "cull_alpha_thresh":       0.005,
        "use_scale_regularization": True,
    },
}


class SplatWorker(BaseWorker):

    job_type = JobType.SPLAT

    async def process(self, job: Job, workdir: Path) -> dict[str, str]:
        # ── Download inputs ───────────────────────────────────────────────────
        frames_dir = self.download_input(job, "frames", workdir)
        sparse_dir = self.download_input(job, "colmap_sparse", workdir)

        log.info(f"Frames: {len(list(frames_dir.glob('*.jpg')))}")
        log.info(f"Sparse model: {sparse_dir}")

        # ── Get hyperparameters ───────────────────────────────────────────────
        plugin  = get_plugin(job.scene_type)
        base_hp = DEFAULT_HYPERPARAMS.get(
            job.scene_type,
            DEFAULT_HYPERPARAMS["default"]
        )
        # Domain plugin can override defaults
        domain_hp = plugin.hyperparameters()
        hp = {**base_hp, **domain_hp}
        # Job-level override (highest priority)
        hp.update(job.metadata.get("hyperparameters", {}))
        n_classes = len(plugin.class_labels())

        log.info(f"Hyperparameters: {hp}")
        log.info(f"Domain classes: {n_classes}")

        # ── Prepare nerfstudio data directory ────────────────────────────────
        ns_data_dir = workdir / "ns_data"
        ns_data_dir.mkdir()

        self._run([
            "ns-process-data", "images",
            "--data",                str(frames_dir),
            "--colmap-model-path",   str(sparse_dir),
            "--output-dir",          str(ns_data_dir),
            "--skip-colmap",
        ])

        transforms_path = ns_data_dir / "transforms.json"
        if not transforms_path.exists():
            raise RuntimeError(
                "ns-process-data failed to produce transforms.json. "
                "Check COLMAP sparse model integrity."
            )

        # ── Inject label maps into transforms.json ────────────────────────────
        # nerfstudio's dataloader will pick up label_map_path per frame
        # self._inject_label_paths(transforms_path, labels_dir)

        # ── Register semantic splatfacto model ────────────────────────────────
        self._register_semantic_model(n_classes)

        # ── Train ─────────────────────────────────────────────────────────────
        output_dir = workdir / "output"
        t0 = time.time()

        train_cmd = self._build_train_cmd(
            ns_data_dir, output_dir, hp, n_classes
        )
        log.info("Starting training...")
        self._run(train_cmd)

        training_time = time.time() - t0
        log.info(f"Training complete in {training_time:.0f}s")

        # ── Export .ply ───────────────────────────────────────────────────────
        config = self._find_config(output_dir)
        export_dir = workdir / "export"
        export_dir.mkdir()

        self._run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config),
            "--output-dir",  str(export_dir),
        ])

        ply_files = list(export_dir.glob("*.ply"))
        if not ply_files:
            raise RuntimeError("Export produced no .ply file")
        ply_path = ply_files[0]

        # ── Compute quality metrics ───────────────────────────────────────────
        metrics = self._extract_training_metrics(config)
        n_gaussians = self._count_gaussians(ply_path)

        metadata = {
            "n_gaussians":      n_gaussians,
            "training_time_s":  round(training_time, 1),
            "hyperparameters":  hp,
            "n_classes":        n_classes,
            "ply_size_mb":      round(ply_path.stat().st_size / 1e6, 1),
            "quality_metrics":  metrics,
            "scene_type":       job.scene_type,
        }

        log.info(
            f"Splat: {n_gaussians:,} Gaussians, "
            f"{metadata['ply_size_mb']:.1f}MB, "
            f"PSNR={metrics.get('psnr', 'N/A')}"
        )

        # ── Upload outputs ────────────────────────────────────────────────────
        splat_uri = self.upload_output(
            job, ply_path, "splat", "splat.ply"
        )
        meta_uri = self.upload_output_json(
            job, metadata, "splat", "metadata.json"
        )

        return {
            "splat":          splat_uri,
            "splat_metadata": meta_uri,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _inject_label_paths(self, transforms_path: Path, labels_dir: Path):
        """
        Add label_map_path to each frame entry in transforms.json.
        nerfstudio's dataloader uses this for semantic supervision.
        """
        with open(transforms_path) as f:
            transforms = json.load(f)

        for frame in transforms["frames"]:
            # frame["file_path"] is like "images/frame_00001.jpg"
            stem = Path(frame["file_path"]).stem
            label_path = labels_dir / f"{stem}_labels.npy"
            if label_path.exists():
                frame["label_map_path"] = str(label_path)

        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

    def _register_semantic_model(self, n_classes: int):
        """
        Register SemanticSplatfacto with nerfstudio's model registry.
        This allows ns-train to use it by name.
        """
        # The semantic model is defined in this worker's src directory
        # nerfstudio discovers it via entry points or direct import
        semantic_model_path = Path(__file__).parent / "semantic_splatfacto.py"
        if not semantic_model_path.exists():
            log.warning(
                "semantic_splatfacto.py not found — using standard splatfacto. "
                "Semantic features will not be trained."
            )

    def _build_train_cmd(
        self,
        data_dir:   Path,
        output_dir: Path,
        hp:         dict,
        n_classes:  int,
    ) -> list[str]:
        # Try semantic model first, fall back to standard splatfacto
        model = "splatfacto"  # TODO: "semantic-splatfacto" once registered

        cmd = [
            "ns-train", model,
            "--data",            str(data_dir),
            "--output-dir",      str(output_dir),
            "--max-num-iterations", str(hp["max_num_iterations"]),
            "--steps-per-save",  "2000",
            "--pipeline.model.num-downscales",
                str(hp.get("num_downscales", 1)),
            "--pipeline.model.cull-alpha-thresh",
                str(hp.get("cull_alpha_thresh", 0.005)),
            "--pipeline.model.use-scale-regularization",
                str(hp.get("use_scale_regularization", True)).lower(),
        ]
        return cmd

    def _find_config(self, output_dir: Path) -> Path:
        configs = sorted(output_dir.rglob("config.yml"))
        if not configs:
            raise RuntimeError(
                f"No config.yml found in {output_dir}. "
                "Did training complete successfully?"
            )
        return configs[-1]

    def _extract_training_metrics(self, config_path: Path) -> dict:
        """
        Parse nerfstudio's training output for quality metrics.
        Looks for the latest eval metrics in the output directory.
        """
        try:
            eval_dir = config_path.parent / "nerfstudio_models"
            # nerfstudio writes metrics to event files
            # Parse the most recent eval output
            metric_files = list(eval_dir.glob("*metrics*.json"))
            if metric_files:
                with open(sorted(metric_files)[-1]) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _count_gaussians(self, ply_path: Path) -> int:
        """Count Gaussians in .ply by reading header."""
        try:
            with open(ply_path, "rb") as f:
                header = b""
                while b"end_header" not in header:
                    header += f.read(1024)
                # Parse "element vertex N" from header
                for line in header.decode("ascii", errors="ignore").split("\n"):
                    if line.startswith("element vertex"):
                        return int(line.split()[-1])
        except Exception:
            pass
        return 0

    def _run(self, cmd: list[str]):
        log.info(f"Running: {' '.join(cmd[:4])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {cmd[0]}\n"
                f"stderr: {result.stderr[-3000:]}"
            )


