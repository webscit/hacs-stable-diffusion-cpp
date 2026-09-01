"""Runtime data for the Stable Diffusion cpp integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .api import StableDiffusionCppClient


@dataclass
class SDCppData:
    """Data stored in a config entry's runtime_data.

    No coordinator: ai_task image generation is an on-demand RPC per call,
    not a data source to poll.
    """

    client: StableDiffusionCppClient


type SDCppConfigEntry = ConfigEntry[SDCppData]
