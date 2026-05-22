# infra/runpod/handler.py

from __future__ import annotations

import asyncio
import urllib.request
import uuid

from pathlib import Path

import runpod

from schemas.jobs import Job, JobType

from workers.frame_extraction.src.worker import FrameExtractionWorker
from workers.colmap.src.worker import ColmapWorker
from workers.splat.src.worker import SplatWorker


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


async def run_pipeline(job_input: dict):

    scene_type = job_input.get("scene_type", "farm_aerial")
    video_url = job_input["input_video_url"]

    scene_id = str(uuid.uuid4())
    snapshot_id = "snapshot_001"

    # ---------------- download input ----------------

    storage_input_dir = Path("storage/inputs")
    storage_input_dir.mkdir(parents=True, exist_ok=True)

    local_video = storage_input_dir / "input.mp4"

    download_file(video_url, local_video)

    # ---------------- frame extraction ----------------

    frame_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.FRAME_EXTRACTION,
        scene_id=scene_id,
        scene_type=scene_type,
        snapshot_id=snapshot_id,
        inputs={
            "video": "local://inputs/input.mp4"
        },
    )

    frame_result = await FrameExtractionWorker().run_job(frame_job)

    if frame_result.status.value != "completed":
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "error": frame_result.error,
        }

    # ---------------- colmap ----------------

    colmap_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.COLMAP,
        scene_id=scene_id,
        scene_type=scene_type,
        snapshot_id=snapshot_id,
        inputs={
            "frames": frame_result.outputs["frames"]
        },
    )

    colmap_result = await ColmapWorker().run_job(colmap_job)

    if colmap_result.status.value != "completed":
        return {
            "status": "failed",
            "stage": "colmap",
            "error": colmap_result.error,
        }

    # ---------------- splat ----------------

    splat_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.SPLAT,
        scene_id=scene_id,
        scene_type=scene_type,
        snapshot_id=snapshot_id,
        inputs={
            "frames": frame_result.outputs["frames"],
            "colmap_sparse": colmap_result.outputs["colmap_sparse"],
        },
        metadata={
            "hyperparameters": {
                "max_num_iterations": 3000
            }
        }
    )

    splat_result = await SplatWorker().run_job(splat_job)

    if splat_result.status.value != "completed":
        return {
            "status": "failed",
            "stage": "splat",
            "error": splat_result.error,
        }

    return {
        "status": "completed",
        "scene_id": scene_id,
        "outputs": splat_result.outputs,
    }


def handler(event):
    return asyncio.run(run_pipeline(event["input"]))


runpod.serverless.start({
    "handler": handler
})