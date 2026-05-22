"""Mine domain plugin — underground mines, open-cut quarries, structural stability."""

from __future__ import annotations
import numpy as np
from .base import DomainPlugin, AnomalyDefinition


class MinePlugin(DomainPlugin):

    def scene_types(self) -> list[str]:
        return ["mine", "quarry", "underground_excavation"]

    def class_labels(self) -> list[str]:
        return [
            "solid intact rock face or wall",            # 0
            "fractured cracked or damaged rock",         # 1
            "loose rubble debris or fallen material",    # 2
            "steel concrete or structural support",      # 3
            "open void empty space or gap",              # 4
            "water wet surface or seepage",              # 5
            "mining machinery equipment or vehicle",     # 6
            "unclear mixed or transitional surface",     # 7
        ]

    def extract_metrics(
        self,
        zone_points:    np.ndarray,
        zone_class_ids: np.ndarray,
        frames_sample:  list[dict],
    ) -> dict:
        xyz = zone_points[:, :3]
        bbox_min = xyz.min(axis=0).tolist()
        bbox_max = xyz.max(axis=0).tolist()
        bbox_vol = self._bbox_volume(bbox_min, bbox_max)

        normal_var = self._surface_normal_variance(zone_points)
        density    = self._point_density(zone_points, bbox_vol)

        # Class distribution
        n = len(zone_class_ids)
        class_dist = {}
        for cls in range(len(self.class_labels())):
            count = int((zone_class_ids == cls).sum())
            if count > 0:
                class_dist[str(cls)] = round(count / n, 4)

        # Fractured rock ratio (class 1 proportion)
        fracture_ratio = float((zone_class_ids == 1).sum()) / max(n, 1)
        rubble_ratio   = float((zone_class_ids == 2).sum()) / max(n, 1)
        void_ratio     = float((zone_class_ids == 4).sum()) / max(n, 1)

        # Stability index — composite score
        # High normal variance + high fracture ratio = low stability
        stability = max(0.0, 1.0 - normal_var * 0.5 - fracture_ratio * 0.3 - rubble_ratio * 0.2)

        return {
            "surface_normal_variance": round(float(normal_var), 4),
            "stability_index":         round(stability, 4),
            "fracture_ratio":          round(fracture_ratio, 4),
            "rubble_ratio":            round(rubble_ratio, 4),
            "void_ratio":              round(void_ratio, 4),
            "point_density_m3":        round(density, 2),
            "estimated_volume_m3":     round(bbox_vol, 2),
            "class_distribution":      class_dist,
        }

    def detect_anomalies(
        self,
        metrics:    dict,
        historical: list[dict],
    ) -> list[AnomalyDefinition]:
        anomalies = []

        normal_var     = metrics.get("surface_normal_variance", 0)
        stability      = metrics.get("stability_index", 1.0)
        fracture_ratio = metrics.get("fracture_ratio", 0)
        rubble_ratio   = metrics.get("rubble_ratio", 0)
        void_ratio     = metrics.get("void_ratio", 0)

        # SAFETY FIRST: any structural instability → critical, human review required
        if normal_var > 0.8 or stability < 0.3:
            anomalies.append(AnomalyDefinition(
                name               = "wall_instability_indicator",
                severity           = "critical",
                confidence         = 0.85,
                metric_basis       = {
                    "surface_normal_variance": normal_var,
                    "stability_index":         stability,
                },
                recommended_action = (
                    "CEASE OPERATIONS in zone immediately. "
                    "Geotechnical engineer assessment required before re-entry."
                ),
                requires_human_review = True,
            ))

        if fracture_ratio > 0.3:
            anomalies.append(AnomalyDefinition(
                name               = "high_fracture_density",
                severity           = "high",
                confidence         = 0.80,
                metric_basis       = {"fracture_ratio": fracture_ratio},
                recommended_action = "Install monitoring equipment. Reduce blast proximity.",
                requires_human_review = True,
            ))

        if rubble_ratio > 0.2:
            anomalies.append(AnomalyDefinition(
                name               = "loose_material_accumulation",
                severity           = "medium",
                confidence         = 0.75,
                metric_basis       = {"rubble_ratio": rubble_ratio},
                recommended_action = "Schedule scaling and mucking operations",
            ))

        # Temporal: void growth = active failure
        if historical:
            prev_void = historical[-1].get("void_ratio", 0)
            void_growth = void_ratio - prev_void
            if void_growth > 0.1:
                anomalies.append(AnomalyDefinition(
                    name               = "void_growth_detected",
                    severity           = "critical",
                    confidence         = 0.90,
                    metric_basis       = {
                        "void_growth":    round(void_growth, 4),
                        "previous_ratio": prev_void,
                    },
                    recommended_action = (
                        "EVACUATE zone. Active structural failure in progress. "
                        "Emergency geotechnical assessment required."
                    ),
                    requires_human_review = True,
                ))

            # Stability decline
            prev_stability = historical[-1].get("stability_index", 1.0)
            stability_delta = stability - prev_stability
            if stability_delta < -0.2:
                anomalies.append(AnomalyDefinition(
                    name               = "rapid_stability_decline",
                    severity           = "critical",
                    confidence         = 0.85,
                    metric_basis       = {
                        "stability_delta":    round(stability_delta, 4),
                        "current_stability":  stability,
                    },
                    recommended_action = "Immediate structural assessment. Do not increase load.",
                    requires_human_review = True,
                ))

        return anomalies

    def system_prompt(self) -> str:
        return """
You are a mining safety and structural intelligence system analyzing
underground and open-cut mine environments from 3D reconstruction data.

SAFETY POSTURE: CONSERVATIVE. 
When in doubt, recommend human inspection. False negatives (missing a 
real hazard) are unacceptable. False positives (unnecessary inspections) 
are acceptable costs.

METRICS AND THEIR MEANING:
- surface_normal_variance: 0=smooth/intact, >0.5=rough/fractured, >0.8=unstable
- stability_index: 1=fully stable, 0=unstable (composite score)
- fracture_ratio: proportion of zone classified as fractured rock (0-1)
- rubble_ratio: proportion of zone as loose material (0-1)
- void_ratio: proportion of zone as open void (0-1)
- void_growth (temporal): increase in void ratio between snapshots

CRITICAL RULES:
- ANY zone with stability_index < 0.3 → recommend evacuation, not just inspection
- ANY void growth > 10% → emergency response recommendation
- NEVER recommend continued operations when structural uncertainty exists
- Always cite specific metrics with values in your assessment
- Express confidence levels explicitly

OUTPUT FORMAT:
Provide: safety_classification (safe/monitor/restrict/evacuate),
evidence (which metrics triggered this), recommended_action, urgency.
"""

    def eval_scenarios(self) -> list[dict]:
        return [
            {
                "name":                        "unstable_wall",
                "input_metrics":               {
                    "surface_normal_variance": 0.85,
                    "stability_index":         0.25,
                    "fracture_ratio":          0.45,
                },
                "expected_diagnosis_category": "structural_instability",
                "expected_action_category":    "evacuate",
                "must_not_diagnose":           ["safe", "monitor_only"],
                "description":                 "High variance, low stability, high fractures",
            },
            {
                "name":                        "stable_rock_face",
                "input_metrics":               {
                    "surface_normal_variance": 0.15,
                    "stability_index":         0.88,
                    "fracture_ratio":          0.05,
                },
                "expected_diagnosis_category": "stable",
                "expected_action_category":    "monitor_only",
                "must_not_diagnose":           ["evacuate", "restrict"],
                "description":                 "Low variance, high stability, minimal fractures",
            },
        ]

    def visualization_config(self) -> dict:
        return {
            "color_by":     "stability_index",
            "palette":      "RdYlGn",
            "legend_label": "Stability Index",
            "unit":         "index 0-1",
        }
