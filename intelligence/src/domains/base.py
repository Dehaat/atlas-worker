"""
Domain plugin base class.
Every domain (farm, mine, tunnel, bridge, etc.) implements this.
The intelligence layer is the only thing that changes per domain.
Reconstruction pipeline is identical for all domains.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ZoneMetrics:
    """
    Base metrics every domain produces per zone.
    Domain plugins extend this with domain-specific fields
    by returning a dict that includes these base fields plus their own.
    """
    zone_id:     str
    scene_id:    str
    snapshot_id: str
    centroid:    list[float]        # [x, y, z] in COLMAP world space
    bbox_min:    list[float]
    bbox_max:    list[float]
    point_count: int
    avg_rgb:     list[float]        # [r, g, b] normalized 0-1

    # Populated by analytics worker after metric extraction
    domain_metrics: dict = field(default_factory=dict)
    anomalies:      list = field(default_factory=list)


@dataclass
class AnomalyDefinition:
    name:               str
    severity:           str     # low | medium | high | critical
    confidence:         float   # 0-1
    metric_basis:       dict    # which metrics triggered this
    recommended_action: str
    requires_human_review: bool = False


class DomainPlugin(ABC):
    """
    Implement this for each domain.
    The system calls these methods in order:
      1. class_labels()        → used by auto_label worker
      2. extract_metrics()     → used by analytics worker
      3. detect_anomalies()    → used by analytics worker
      4. system_prompt()       → used by intelligence worker (LLM)
      5. eval_scenarios()      → used by eval framework
    """

    @abstractmethod
    def scene_types(self) -> list[str]:
        """Which scene type strings this plugin handles."""
        ...

    @abstractmethod
    def class_labels(self) -> list[str]:
        """
        Human-readable class labels for CLIP zero-shot classification.
        Write these as descriptive phrases, not single words.
        CLIP performs better with natural language descriptions.

        Example: "healthy green crop vegetation" not "crop"
        """
        ...

    @abstractmethod
    def extract_metrics(
        self,
        zone_points:   np.ndarray,   # (N, 6): x, y, z, r, g, b
        zone_class_ids: np.ndarray,  # (N,) int: per-point class from splat
        frames_sample:  list[dict],  # [{path, R, t, K}] for projection
    ) -> dict:
        """
        Compute domain-specific metrics for a zone.
        Returns a flat dict of metric_name → value.
        All values must be JSON-serializable (float, int, str, list).
        """
        ...

    @abstractmethod
    def detect_anomalies(
        self,
        metrics:    dict,
        historical: list[dict],   # previous snapshots' metrics, oldest first
    ) -> list[AnomalyDefinition]:
        """
        Rule-based anomaly detection.
        Runs before the LLM — catches obvious cases deterministically.
        Fast, cheap, auditable.
        """
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """
        Domain-specific system prompt for the intelligence agent.
        Should include:
          - Role and domain context
          - What each metric means and its normal range
          - Known failure modes to avoid
          - Safety posture (conservative vs liberal)
          - Output format expectations
        """
        ...

    @abstractmethod
    def eval_scenarios(self) -> list[dict]:
        """
        Known input→output test cases for eval framework.
        Each scenario:
          {
            "name": str,
            "input_metrics": dict,
            "expected_diagnosis_category": str,
            "expected_action_category": str,
            "must_not_diagnose": list[str],
            "description": str,
          }
        """
        ...

    def hyperparameters(self) -> dict:
        """
        Override default 3DGS training hyperparameters for this scene type.
        Return empty dict to use defaults.
        """
        return {}

    def visualization_config(self) -> dict:
        """
        How to color zones in the viewer.
        """
        return {
            "color_by":     "health_score",
            "palette":      "RdYlGn",
            "legend_label": "Zone Score",
            "unit":         "score",
        }

    def requires_multispectral(self) -> bool:
        """Does this domain need multispectral imagery for accurate metrics?"""
        return False

    def requires_thermal(self) -> bool:
        """Does this domain benefit from thermal imagery?"""
        return False

    # ── Shared utilities available to all plugins ─────────────────────────────

    def _grvi(self, avg_rgb: list[float]) -> float:
        """Green-Red Vegetation Index — RGB proxy for crop health."""
        r, g, _ = avg_rgb
        return float((g - r) / (g + r + 1e-6))

    def _point_density(self, points: np.ndarray, volume: float) -> float:
        """Points per cubic meter."""
        return float(len(points) / max(volume, 1e-6))

    def _surface_normal_variance(self, points: np.ndarray) -> float:
        """
        Variance of surface normals estimated from point neighbors.
        High variance = rough/fractured surface.
        Low variance = smooth/intact surface.
        """
        if len(points) < 10:
            return 0.0
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=10).fit(points[:, :3])
            _, indices = nn.kneighbors(points[:, :3])

            variances = []
            for i, neighbors in enumerate(indices):
                pts = points[neighbors, :3]
                centered = pts - pts.mean(axis=0)
                _, _, vt = np.linalg.svd(centered)
                normal = vt[-1]  # smallest singular vector = normal
                variances.append(float(np.linalg.norm(normal)))

            return float(np.var(variances))
        except Exception:
            return 0.0

    def _bbox_volume(self, bbox_min: list, bbox_max: list) -> float:
        """Volume of axis-aligned bounding box in cubic units."""
        dims = [abs(bbox_max[i] - bbox_min[i]) for i in range(3)]
        return dims[0] * dims[1] * dims[2]
