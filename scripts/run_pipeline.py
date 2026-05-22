from __future__ import annotations

import argparse
import asyncio
import shutil
import uuid
from pathlib import Path

from jobs import Job, JobType
from workers.frame_extraction.src.worker import FrameExtractionWorker
from workers.colmap.src.worker import ColmapWorker
from workers.splat.src.worker import SplatWorker


SCENE_ID = "farm_test"
SNAPSHOT_ID = "snapshot_001"
DEFAULT_VIDEO = Path("storage/inputs/sample.mp4")


def stage_video(video_path: Path) -> str:
    """Copy user video into storage/inputs and return its local:// URI."""
    src = video_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}")

    dst = DEFAULT_VIDEO.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copy2(src, dst)
        print(f"Staged video: {src} -> {dst}")
    else:
        print(f"Using video: {dst}")

    return "local://inputs/sample.mp4"


async def main(video_uri: str):

    # ---------------- frame extraction ----------------

    frame_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.FRAME_EXTRACTION,
        scene_id=SCENE_ID,
        scene_type="farm_aerial",
        snapshot_id=SNAPSHOT_ID,
        inputs={
            "video": video_uri,
        },
    )

    frame_worker = FrameExtractionWorker()
    frame_result = await frame_worker.run_job(frame_job)

    print(frame_result.to_json())

    if frame_result.status.value != "completed":
        return

    # ---------------- colmap ----------------

    colmap_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.COLMAP,
        scene_id=SCENE_ID,
        scene_type="farm_aerial",
        snapshot_id=SNAPSHOT_ID,
        inputs={
            "frames": frame_result.outputs["frames"]
        },
    )

    colmap_worker = ColmapWorker()
    colmap_result = await colmap_worker.run_job(colmap_job)

    print(colmap_result.to_json())

    if colmap_result.status.value != "completed":
        return

    # ---------------- splat ----------------

    splat_job = Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.SPLAT,
        scene_id=SCENE_ID,
        scene_type="farm_aerial",
        snapshot_id=SNAPSHOT_ID,
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

    splat_worker = SplatWorker()
    splat_result = await splat_worker.run_job(splat_job)

    print(splat_result.to_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 reconstruction pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Path to input .mp4 (copied to storage/inputs/sample.mp4)",
    )
    args = parser.parse_args()
    asyncio.run(main(stage_video(args.video)))