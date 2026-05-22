"""Farm domain plugin — agriculture, crop health, irrigation analysis."""

from __future__ import annotations
import numpy as np
from .base import DomainPlugin, AnomalyDefinition


class FarmPlugin(DomainPlugin):

    def scene_types(self) -> list[str]:
        return ["farm_aerial", "orchard", "vineyard", "farm"]

    def class_labels(self) -> list[str]:
        return [
            "healthy green crop vegetation and leaves",  # 0
            "stressed yellowing or wilting crop",        # 1
            "bare dry exposed soil or dirt",             # 2
            "water irrigation channel or pond",          # 3
            "farm road path or track",                   # 4
            "farm building shed or structure",           # 5
            "dead brown dry or harvested vegetation",    # 6
            "mixed unclear or transitional area",        # 7
        ]

    def extract_metrics(
        self,
        zone_points:    np.ndarray,
        zone_class_ids: np.ndarray,
        frames_sample:  list[dict],
    ) -> dict:
        avg_rgb = zone_points[:, 3:6].mean(axis=0) / 255.0
        grvi    = self._grvi(avg_rgb.tolist())
        health  = min(max((grvi + 1) / 2, 0.0), 1.0)

        # Class distribution in zone
        n = len(zone_class_ids)
        class_dist = {}
        for cls in range(len(self.class_labels())):
            count = int((zone_class_ids == cls).sum())
            if count > 0:
                class_dist[str(cls)] = round(count / n, 4)

        # Dominant class
        dominant_class = int(np.bincount(
            zone_class_ids[zone_class_ids >= 0],
            minlength=len(self.class_labels())
        ).argmax()) if (zone_class_ids >= 0).any() else -1

        # Spatial coverage estimate
        bbox_vol = self._bbox_volume(
            zone_points[:, :3].min(axis=0).tolist(),
            zone_points[:, :3].max(axis=0).tolist()
        )

        return {
            "grvi":              round(grvi, 4),
            "health_score":      round(health, 4),
            "avg_rgb":           avg_rgb.tolist(),
            "dominant_class":    dominant_class,
            "class_distribution": class_dist,
            "estimated_area_m2": round(bbox_vol ** (2/3), 2),  # rough surface area
            "point_density":     round(float(len(zone_points)) / max(bbox_vol, 0.01), 2),
        }

    def detect_anomalies(
        self,
        metrics:    dict,
        historical: list[dict],
    ) -> list[AnomalyDefinition]:
        anomalies = []
        grvi   = metrics.get("grvi", 0)
        health = metrics.get("health_score", 0.5)

        # Current state anomalies
        if grvi < -0.3:
            anomalies.append(AnomalyDefinition(
                name               = "severe_vegetation_stress",
                severity           = "high",
                confidence         = 0.90,
                metric_basis       = {"grvi": grvi},
                recommended_action = "Inspect irrigation supply and soil moisture immediately",
                requires_human_review = True,
            ))
        elif grvi < -0.1:
            anomalies.append(AnomalyDefinition(
                name               = "moderate_vegetation_stress",
                severity           = "medium",
                confidence         = 0.75,
                metric_basis       = {"grvi": grvi},
                recommended_action = "Monitor closely, check irrigation schedule",
            ))

        # Temporal anomalies — require historical data
        if historical:
            prev = historical[-1]
            prev_grvi   = prev.get("grvi", 0)
            grvi_delta  = grvi - prev_grvi

            if grvi_delta < -0.25:
                anomalies.append(AnomalyDefinition(
                    name               = "rapid_vegetation_decline",
                    severity           = "critical",
                    confidence         = 0.88,
                    metric_basis       = {"grvi_delta": round(grvi_delta, 4), "prev_grvi": prev_grvi},
                    recommended_action = "Immediate field inspection required — rapid decline detected",
                    requires_human_review = True,
                ))
            elif grvi_delta < -0.10:
                anomalies.append(AnomalyDefinition(
                    name               = "gradual_vegetation_decline",
                    severity           = "medium",
                    confidence         = 0.70,
                    metric_basis       = {"grvi_delta": round(grvi_delta, 4)},
                    recommended_action = "Schedule field inspection within 48 hours",
                ))

        return anomalies

    def system_prompt(self) -> str:
        return """
You are a precision agriculture intelligence system analyzing crop health
from aerial imagery reconstruction data.

METRICS AND THEIR MEANING:
- grvi (Green-Red Vegetation Index): healthy > 0.2, moderate 0.0-0.2,
  stressed -0.2 to 0.0, severely stressed < -0.2
- health_score: normalized 0-1 from grvi (1 = optimal)
- dominant_class: 0=healthy crop, 1=stressed crop, 2=bare soil,
  3=water, 4=road, 5=structure, 6=dead vegetation, 7=mixed

DIAGNOSTIC RULES:
- Drought stress: gradual decline across large contiguous zones
- Irrigation failure: sharp boundaries, zone-bounded decline
- Pest/disease: irregular patches with healthy neighbors
- Seasonal variation: compare against same period in prior year

IMPORTANT CONSTRAINTS:
- Never diagnose disease without temporal context
- Always check if baseline snapshot exists before flagging decline
- Distinguish drought (gradual, regional) from irrigation failure (sharp, bounded)
- Express confidence levels explicitly
- If baseline unavailable, state this and recommend establishing one

OUTPUT FORMAT:
Provide structured assessment with: diagnosis, confidence, evidence, recommendation, priority.
"""

    def eval_scenarios(self) -> list[dict]:
        return [
            {
                "name":                        "blocked_irrigation_pipe",
                "input_metrics":               {"grvi": -0.31, "health_score": 0.35, "dominant_class": 1},
                "expected_diagnosis_category": "irrigation_failure",
                "expected_action_category":    "physical_inspection",
                "must_not_diagnose":           ["seasonal_variation", "harvest"],
                "description":                 "Sharp zone boundary, neighbors healthy, no recent rain",
            },
            {
                "name":                        "healthy_field",
                "input_metrics":               {"grvi": 0.42, "health_score": 0.71, "dominant_class": 0},
                "expected_diagnosis_category": "normal",
                "expected_action_category":    "monitor_only",
                "must_not_diagnose":           ["stress", "failure", "disease"],
                "description":                 "All metrics within healthy range",
            },
            {
                "name":                        "seasonal_variation",
                "input_metrics":               {"grvi": 0.15, "health_score": 0.58, "dominant_class": 0},
                "expected_diagnosis_category": "normal_seasonal_variation",
                "expected_action_category":    "monitor_only",
                "must_not_diagnose":           ["stress", "disease"],
                "description":                 "Late season, same grvi as last year same period",
            },
        ]

    def hyperparameters(self) -> dict:
        return {
            "max_num_iterations":     30000,
            "densify_grad_threshold": 0.0002,
            "num_downscales":         2,
        }

    def visualization_config(self) -> dict:
        return {
            "color_by":     "health_score",
            "palette":      "RdYlGn",
            "legend_label": "Crop Health",
            "unit":         "GRVI score",
        }

    def requires_multispectral(self) -> bool:
        return True  # real NDVI needs NIR band
