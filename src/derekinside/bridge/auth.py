"""
derekinside — Simple token-based auth for bridge endpoints.

Uses a shared token configured in config.yaml or env var.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    enabled: bool = False
    token: str = ""
    header: str = "X-DEREINSIDE-TOKEN"


class Auth:
    """Token-based auth for derekinside bridge."""

    def __init__(self, config: Optional[AuthConfig] = None):
        self._config = config or AuthConfig()
        # Fallback to env var
        if not self._config.token:
            self._config.token = os.environ.get("DEREINSIDE_TOKEN", "")

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.token)

    def check(self, token: str) -> bool:
        """Check if token is valid (constant-time comparison)."""
        if not self.enabled:
            return True
        if not token:
            return False
        return hmac.compare_digest(token, self._config.token)
