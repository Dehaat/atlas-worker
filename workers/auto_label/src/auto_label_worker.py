"""
Auto-Label Worker

Input:  frames artifact, colmap_sparse artifact (for camera poses)
Output: label_maps artifact (per-frame .npy semantic label maps)
        label_metadata JSON

Uses SAM2 for segmentation + CLIP for zero-shot classification.
No human annotation required.
Labels are used as supervision signal for semantic splat training.
"""

from __future__ import annotations
import asyncio
import json
import numpy as np
from pathlib import Path

import sys

from sdk import BaseWorker, get_logger
from jobs import Job, JobType
from intelligence.src.domains.registry import get_plugin

log = get_logger("atlas.worker.auto_label")


class AutoLabelWorker(BaseWorker):

    job_type = JobType.AUTO_LABEL

    async def process(self, job: Job, workdir: Path) -> dict[str, str]:
        # ── Download inputs ───────────────────────────────────────────────────
        frames_dir = self.download_input(job, "frames", workdir)
        frames     = sorted(frames_dir.glob("*.jpg"))
        log.info(f"Labeling {len(frames)} frames")

        # ── Get domain class labels ───────────────────────────────────────────
        plugin       = get_plugin(job.scene_type)
        class_labels = plugin.class_labels()
        n_classes    = len(class_labels)
        log.info(f"Domain: {job.scene_type}, {n_classes} classes: {class_labels}")

        # ── Load models ───────────────────────────────────────────────────────
        labeler = self._build_labeler(class_labels)

        # ── Label frames ─────────────────────────────────────────────────────
        labels_dir = workdir / "label_maps"
        labels_dir.mkdir()

        label_stats: list[dict] = []
        for i, frame_path in enumerate(frames):
            if i % 10 == 0:
                log.info(f"Labeling frame {i+1}/{len(frames)}")

            label_map, stats = labeler.label_frame(frame_path)
            out_path = labels_dir / f"{frame_path.stem}_labels.npy"
            np.save(str(out_path), label_map)
            label_stats.append(stats)

        # ── Compute aggregate stats ───────────────────────────────────────────
        class_coverage = self._aggregate_coverage(label_stats, n_classes)
        unlabeled_rate = np.mean([s["unlabeled_rate"] for s in label_stats])

        metadata = {
            "n_frames":       len(frames),
            "n_classes":      n_classes,
            "class_labels":   class_labels,
            "class_coverage": class_coverage,
            "unlabeled_rate": round(float(unlabeled_rate), 4),
            "model":          labeler.model_info(),
        }

        log.info(
            f"Labeling complete. Unlabeled rate: {unlabeled_rate:.1%}. "
            f"Coverage: {class_coverage}"
        )

        # ── Upload outputs ────────────────────────────────────────────────────
        labels_uri = self.upload_output(
            job, labels_dir, "label_maps", ""
        )
        meta_uri = self.upload_output_json(
            job, metadata, "label_maps", "metadata.json"
        )

        return {
            "label_maps":          labels_uri,
            "label_maps_metadata": meta_uri,
        }

    def _build_labeler(self, class_labels: list[str]) -> "Labeler":
        return Labeler(class_labels)

    def _aggregate_coverage(
        self,
        stats:     list[dict],
        n_classes: int
    ) -> dict[str, float]:
        """Compute average coverage per class across all frames."""
        totals = {i: 0.0 for i in range(n_classes)}
        for s in stats:
            for class_id, coverage in s.get("class_coverage", {}).items():
                totals[int(class_id)] += coverage
        n = len(stats) or 1
        return {str(k): round(v / n, 4) for k, v in totals.items()}


class Labeler:
    """
    SAM2 segmentation + CLIP zero-shot classification.
    Generates per-pixel class labels from RGB frames.
    """

    CLIP_CONFIDENCE_THRESHOLD = 0.20  # ignore low-confidence classifications

    def __init__(self, class_labels: list[str]):
        self.class_labels = class_labels
        self._sam2       = None
        self._clip_model  = None
        self._clip_preprocess = None
        self._class_embeddings = None
        self._load_models()

    def _load_models(self):
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading models on {self._device}")

        # SAM2
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_checkpoint = "sam2_hiera_large.pt"
            model_cfg       = "sam2_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self._device)
            self._sam2 = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side       = 32,
                pred_iou_thresh       = 0.86,
                stability_score_thresh = 0.92,
                min_mask_region_area  = 200,
            )
            log.info("SAM2 loaded")
        except Exception as e:
            log.warning(f"SAM2 not available: {e}. Falling back to grid labeling.")
            self._sam2 = None

        # CLIP
        try:
            import clip
            import torch
            self._clip_model, self._clip_preprocess = clip.load(
                "ViT-B/32", device=self._device
            )
            tokens = clip.tokenize(self.class_labels).to(self._device)
            with torch.no_grad():
                emb = self._clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            self._class_embeddings = emb
            log.info("CLIP loaded")
        except Exception as e:
            log.warning(f"CLIP not available: {e}. Labels will be unlabeled.")
            self._clip_model = None

    def label_frame(self, image_path: Path) -> tuple[np.ndarray, dict]:
        """
        Label a single frame.
        Returns: (H, W) int32 label map (-1 = unlabeled) + stats dict.
        """
        import cv2
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W      = image_rgb.shape[:2]
        label_map = np.full((H, W), -1, dtype=np.int32)

        if self._sam2 is None or self._clip_model is None:
            # Fallback: divide image into a 4x4 grid, classify each cell
            label_map = self._grid_label(image_rgb)
        else:
            label_map = self._sam2_clip_label(image_rgb, label_map)

        # Stats
        unlabeled     = (label_map == -1).sum()
        unlabeled_rate = float(unlabeled) / (H * W)
        class_coverage = {}
        for cls in range(len(self.class_labels)):
            coverage = float((label_map == cls).sum()) / (H * W)
            if coverage > 0:
                class_coverage[cls] = round(coverage, 4)

        return label_map, {
            "frame":          image_path.name,
            "unlabeled_rate": round(unlabeled_rate, 4),
            "class_coverage": class_coverage,
        }

    def _sam2_clip_label(
        self, image_rgb: np.ndarray, label_map: np.ndarray
    ) -> np.ndarray:
        import torch
        from PIL import Image

        # SAM2: generate all segments
        masks = self._sam2.generate(image_rgb)

        # Sort by area descending — larger segments first
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        for mask_data in masks:
            mask = mask_data["segmentation"]   # (H, W) bool
            if mask.sum() < 100:               # skip tiny segments
                continue

            # Crop the segment bounding box for CLIP
            rows, cols = np.where(mask)
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())

            # Minimum crop size for CLIP
            if (r1 - r0) < 8 or (c1 - c0) < 8:
                continue

            crop = Image.fromarray(image_rgb[r0:r1, c0:c1])

            # CLIP: classify crop
            clip_input = self._clip_preprocess(crop).unsqueeze(0).to(self._device)
            with torch.no_grad():
                img_emb = self._clip_model.encode_image(clip_input)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                similarity = (img_emb @ self._class_embeddings.T).squeeze()
                class_idx  = int(similarity.argmax())
                confidence = float(similarity[class_idx])

            if confidence >= self.CLIP_CONFIDENCE_THRESHOLD:
                label_map[mask] = class_idx

        return label_map

    def _grid_label(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Fallback when SAM2/CLIP unavailable.
        Divides image into grid and classifies each cell.
        Crude but functional.
        """
        import torch
        from PIL import Image

        H, W = image_rgb.shape[:2]
        label_map = np.full((H, W), -1, dtype=np.int32)

        if self._clip_model is None:
            return label_map

        rows, cols = 4, 4
        rh, cw = H // rows, W // cols

        for r in range(rows):
            for c in range(cols):
                r0, r1 = r * rh, (r + 1) * rh
                c0, c1 = c * cw, (c + 1) * cw
                crop = Image.fromarray(image_rgb[r0:r1, c0:c1])
                clip_input = self._clip_preprocess(crop).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    img_emb = self._clip_model.encode_image(clip_input)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    sim = (img_emb @ self._class_embeddings.T).squeeze()
                    cls = int(sim.argmax())
                    conf = float(sim[cls])
                if conf >= self.CLIP_CONFIDENCE_THRESHOLD:
                    label_map[r0:r1, c0:c1] = cls

        return label_map

    def model_info(self) -> dict:
        return {
            "segmentation": "sam2_hiera_large" if self._sam2 else "grid_fallback",
            "classification": "clip_ViT-B/32" if self._clip_model else "none",
            "confidence_threshold": self.CLIP_CONFIDENCE_THRESHOLD,
        }


async def main():
    worker = AutoLabelWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
