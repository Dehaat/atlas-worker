from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class JobType(str, Enum):
    FRAME_EXTRACTION = "frame_extraction"
    COLMAP = "colmap"
    SPLAT = "splat"


class JobStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    job_type: JobType
    scene_id: str
    scene_type: str
    snapshot_id: str
    inputs: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_json(self):
        d = asdict(self)
        d["job_type"] = self.job_type.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str | bytes):
        d = json.loads(data)
        d["job_type"] = JobType(d["job_type"])
        return cls(**d)


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    duration_secs: float = 0.0

    def to_json(self):
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, indent=2)
