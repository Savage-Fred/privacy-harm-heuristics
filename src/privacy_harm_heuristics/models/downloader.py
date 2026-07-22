"""Backblaze B2 model downloader utility — STUBBED for this extraction.

The old repo's version of this module made live Backblaze B2 REST API calls
(via `requests`, not a dependency of this package) to sync trained-model
artifacts from a private B2 bucket. That bucket and the associated
credentials are infrastructure specific to the private old repo and are out
of scope for this practicum artifact (see extraction STOP RULE: "GCS/B2
download paths in models/downloader.py can raise NotImplementedError with a
note"). `validate_credentials()` and `sync_models_from_backblaze()` keep
their original signatures so `models/backend.py`'s lazy import
(`from .downloader import sync_models_from_backblaze`) still resolves; they
raise `NotImplementedError` if actually called.

Environment variables the old implementation used (kept for reference only):
  B2_ACCOUNT_ID / BACKBLAZE_KEY_ID
  B2_APP_KEY / BACKBLAZE_API_KEY
  B2_BUCKET / BACKBLAZE_BUCKET
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class BackblazeCredentialsError(RuntimeError):
    pass


def validate_credentials() -> None:
    raise NotImplementedError(
        "Backblaze B2 model sync was not extracted from the old repo "
        "(private-infra-specific; see models/downloader.py docstring)."
    )


def sync_models_from_backblaze(
    dst_dir: Path = Path("models"), prefix: Optional[str] = None, dry_run: bool = False
) -> list[str]:
    raise NotImplementedError(
        "Backblaze B2 model sync was not extracted from the old repo "
        "(private-infra-specific; see models/downloader.py docstring)."
    )
