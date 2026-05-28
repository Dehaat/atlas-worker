"""
Base worker — stateless transformation unit for Phase 1 pipeline jobs.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from schemas.jobs import Job, JobResult, JobStatus, JobType

from .logging import get_logger
from .storage import StorageClient


class BaseWorker(ABC):
    """Inherit from this and implement ``process(job, workdir)``."""

    job_type: JobType

    def __init__(self):
        self.log = get_logger(f"atlas.worker.{self.job_type.value}")
        self.storage = StorageClient()

    async def run_job(self, job: Job) -> JobResult:
        """Run a single job in a temporary work directory."""
        start = time.time()
        with tempfile.TemporaryDirectory(
            prefix=f"atlas_{job.job_id}_"
        ) as tmp:
            workdir = Path(tmp)
            try:
                outputs = await self.process(job, workdir)
                duration = time.time() - start
                self.log.info(
                    f"Job {job.job_id} completed in {duration:.1f}s "
                    f"with {len(outputs)} outputs"
                )
                return JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    outputs=outputs,
                    duration_secs=duration,
                )
            except Exception as e:
                duration = time.time() - start
                tb = traceback.format_exc()
                self.log.error(f"Job {job.job_id} failed: {e}\n{tb}")
                return JobResult(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error=f"{e}\n{tb}",
                    duration_secs=duration,
                )

    def download_input(
        self, job: Job, artifact_type: str, workdir: Path
    ) -> Path:
        uri = job.inputs.get(artifact_type)
        if not uri:
            raise ValueError(f"Missing input artifact: {artifact_type}")
        return self.storage.download_dir(
            uri, workdir / artifact_type
        )

    def download_input_file(
        self, job: Job, artifact_type: str, local_path: Path
    ) -> Path:
        uri = job.inputs.get(artifact_type)
        if not uri:
            raise ValueError(f"Missing input artifact: {artifact_type}")
        return self.storage.download_file(uri, local_path)

    def upload_output(
        self,
        job: Job,
        local_path: Path,
        artifact_type: str,
        filename: str,
    ) -> str:
        uri = self.storage.artifact_uri(
            job.scene_id, artifact_type, filename
        )
        if local_path.is_dir():
            return self.storage.upload_dir(local_path, uri)
        return self.storage.upload_file(local_path, uri)

    def upload_output_json(
        self,
        job: Job,
        data: dict | list,
        artifact_type: str,
        filename: str,
    ) -> str:
        uri = self.storage.artifact_uri(
            job.scene_id, artifact_type, filename
        )
        return self.storage.upload_json(data, uri)

    @abstractmethod
    async def process(self, job: Job, workdir: Path) -> dict[str, str]:
        """Run worker logic; return artifact_type → local:// URI."""
