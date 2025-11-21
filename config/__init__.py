"""
MAGNET Configuration Package
=============================

Provides configuration for MAGNET naval design research system.
"""

from config.magnet_config import MAGNETSettings, get_default_profile

def get_settings() -> MAGNETSettings:
    """Get current MAGNET settings."""
    profile = get_default_profile()
    return profile.settings

# Legacy alias for compatibility
ARCSettings = MAGNETSettings

def reset_settings_cache():
    """Reset settings cache (no-op for MAGNET but kept for test compatibility)."""
    pass

__all__ = ['get_settings', 'MAGNETSettings', 'get_default_profile', 'ARCSettings', 'reset_settings_cache']
