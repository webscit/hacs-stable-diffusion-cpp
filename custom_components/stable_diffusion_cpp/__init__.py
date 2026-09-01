"""The Stable Diffusion cpp integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from .api import StableDiffusionCppClient
from .const import CONF_BASE_URL
from .data import SDCppConfigEntry, SDCppData

PLATFORMS = (Platform.AI_TASK,)


async def async_setup_entry(hass: HomeAssistant, entry: SDCppConfigEntry) -> bool:
    """Set up Stable Diffusion cpp from a config entry."""
    session = aiohttp_client.async_get_clientsession(hass)
    client = StableDiffusionCppClient(entry.data[CONF_BASE_URL], session)
    entry.runtime_data = SDCppData(client=client)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SDCppConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass: HomeAssistant, entry: SDCppConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)
