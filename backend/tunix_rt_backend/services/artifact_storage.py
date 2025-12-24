"""Artifact storage service (M20).

Handles local filesystem storage for model artifacts using content-addressable storage scheme.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path

from tunix_rt_backend.settings import settings

logger = logging.getLogger(__name__)


class ArtifactStorageService:
    """Service for managing artifact storage on filesystem."""

    def __init__(self, root_path: str | None = None):
        """Initialize storage service.

        Args:
            root_path: Root directory for storage. Defaults to settings.model_registry_path.
        """
        self.root_path = Path(root_path or settings.model_registry_path)
        # Ensure root exists
        if not self.root_path.exists():
            try:
                self.root_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create registry root {self.root_path}: {e}")

    def put_directory(self, src_dir: str) -> tuple[str, str, int]:
        """Store a directory of artifacts.

        Calculates SHA256 of directory content and copies files to
        {root_path}/{sha256}/.

        Args:
            src_dir: Source directory path

        Returns:
            Tuple of (storage_uri, sha256, size_bytes)
        """
        src_path = Path(src_dir).resolve()
        if not src_path.exists() or not src_path.is_dir():
            raise ValueError(f"Source directory not found: {src_dir}")

        # 1. Compute SHA256 and size
        sha256, size_bytes = self._compute_dir_hash(src_path)

        # 2. Determine target path
        target_path = self.root_path / sha256

        # 3. Copy if not exists
        if target_path.exists():
            logger.info(f"Artifact {sha256} already exists. Skipping copy.")
        else:
            try:
                # Copy tree
                shutil.copytree(src_path, target_path, dirs_exist_ok=True)
                logger.info(f"Stored artifact {sha256} ({size_bytes} bytes)")
            except Exception as e:
                # Cleanup partial
                if target_path.exists():
                    shutil.rmtree(target_path)
                raise RuntimeError(f"Failed to store artifact: {e}")

        # 4. Return metadata
        # URI scheme: file://{absolute_path}
        storage_uri = target_path.as_uri()

        return storage_uri, sha256, size_bytes

    def get(self, storage_uri: str) -> Path:
        """Get local path for an artifact URI.

        Args:
            storage_uri: Storage URI (must start with file://)

        Returns:
            Path object to the artifact directory
        """
        if not storage_uri.startswith("file://"):
            # Handle relative paths or just assume it's a path if not URI
            # But for safety we expect what we produce
            pass

        # Convert URI to path
        # Simple parsing for local files
        if storage_uri.startswith("file:///"):
            # absolute path on unix or windows with drive?
            # pathlib.Path(uri) doesn't work directly
            # urllib.parse.urlparse is safer
            from urllib.parse import unquote, urlparse

            parsed = urlparse(storage_uri)
            path_str = unquote(parsed.path)

            # On Windows, /C:/... needs handling
            path = Path(path_str)
            # If path starts with / and it's windows, and follows with Drive letter, strip first /
            if os.name == "nt" and str(path).startswith("\\") and ":" in str(path):
                # This is tricky cross-platform.
                # Let's rely on creating the path object from the string after scheme.
                pass

            # Simpler: just strip file:// scheme if we know it is local
            # But urlparse is better.
            # actually Path.from_uri() isn't standard in old python?

            # Let's re-implement simple stripping for known local scheme
            # This matches how we generated it with as_uri()
            # On Windows as_uri() produces file:///D:/...

            # Path(url2pathname(parsed.path)) is the robust way
            from urllib.request import url2pathname

            path = Path(url2pathname(parsed.path))

        else:
            # Fallback if just a path string
            path = Path(storage_uri)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")

        return path

    def _compute_dir_hash(self, directory: Path) -> tuple[str, int]:
        """Compute SHA256 hash and size of a directory.

        Hash is computed over filenames and content to ensure
        identical structure and content produces same hash.
        """
        sha_hash = hashlib.sha256()
        total_size = 0

        # Walk directory
        # Sort to ensure deterministic order
        paths = sorted(directory.rglob("*"))

        for path in paths:
            if path.is_file():
                # Update hash with relative path
                rel_path = path.relative_to(directory).as_posix()
                sha_hash.update(rel_path.encode("utf-8"))

                # Update hash with content
                # Read in chunks
                try:
                    size = path.stat().st_size
                    total_size += size

                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha_hash.update(chunk)
                except OSError as e:
                    logger.warning(f"Could not read file {path} for hashing: {e}")

        return sha_hash.hexdigest(), total_size
