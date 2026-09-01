"""Base entity for the Stable Diffusion cpp integration."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity

from .const import DOMAIN


class SDCppEntity(Entity):
    """Base entity backed by a stable-diffusion.cpp server."""

    _attr_has_entity_name = True

    def __init__(self, entry_id: str, unique_id_suffix: str) -> None:
        """Initialize the entity.

        `unique_id_suffix` disambiguates entities if a second platform is
        added later; two entities under the same config entry must not
        share a unique_id.
        """
        self._attr_unique_id = f"{entry_id}-{unique_id_suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="Stable Diffusion cpp",
            manufacturer="stable-diffusion.cpp",
        )
