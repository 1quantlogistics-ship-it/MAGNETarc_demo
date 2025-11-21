"""
ARC Configuration Package
=========================

Provides YAML-based configuration loading for agents, models, and consensus.
Uses config/loader.py for configuration management.
"""

from config.loader import ConfigLoader, get_config_loader

try:
    from config.magnet_config import MAGNETSettings, get_default_profile

    def get_settings() -> MAGNETSettings:
        profile = get_default_profile()
        return profile.settings

    __all__ = ['ConfigLoader', 'get_config_loader', 'get_settings', 'MAGNETSettings', 'get_default_profile']
except ImportError:
    __all__ = ['ConfigLoader', 'get_config_loader']
