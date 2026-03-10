"""CAPTCHA module for motion-based human verification."""

from .generator import (
    generate_captcha,
    verify_captcha,
    cleanup_expired_captchas,
)
from .optimized_core import warm_up_jit

__all__ = [
    "generate_captcha",
    "verify_captcha",
    "cleanup_expired_captchas",
    "warm_up_jit",
]
