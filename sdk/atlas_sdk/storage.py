from __future__ import annotations

import json
import shutil
from pathlib import Path


class StorageClient:
    """
    Local filesystem storage backend.

    All artifacts use local:// URIs. Paths resolve under the storage root
    (default: ``storage/`` at repo root).

    URI format::

        local://inputs/sample.mp4
        local://farm_test/frames
        local://farm_test/colmap_sparse
        local://farm_test/splat/splat.ply
    """

    def __init__(self, root: str = "storage"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def artifact_uri(
        self, scene_id: str, artifact_type: str, filename: str = ""
    ) -> str:
        rel = f"{scene_id}/{artifact_type}"
        if filename:
            rel = f"{rel}/{filename}"
        return f"local://{rel}"

    def _resolve(self, uri: str) -> Path:
        if not uri.startswith("local://"):
            raise ValueError(f"Unsupported URI (expected local://): {uri}")
        rel = uri.replace("local://", "", 1)
        return self.root / rel

    def upload_file(self, local_path: str | Path, uri: str) -> str:
        dst = self._resolve(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
        return uri

    def upload_dir(self, local_dir: str | Path, uri: str) -> str:
        dst = self._resolve(uri)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(local_dir, dst)
        return uri

    def upload_json(self, obj: dict | list, uri: str) -> str:
        dst = self._resolve(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(obj, f, indent=2)
        return uri

    def download_file(self, uri: str, local_path: str | Path) -> Path:
        src = self._resolve(uri)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path

    def download_dir(self, uri: str, local_dir: str | Path) -> Path:
        src = self._resolve(uri)
        local_dir = Path(local_dir)
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(src, local_dir)
        return local_dir
