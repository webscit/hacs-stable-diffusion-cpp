"""Tests for the Stable Diffusion cpp integration setup/unload."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.stable_diffusion_cpp.const import CONF_BASE_URL, DOMAIN

BASE_URL = "http://127.0.0.1:1234"
ENTITY_ID = "ai_task.stable_diffusion_cpp_image_generation"


async def test_setup_and_unload_entry(hass: HomeAssistant) -> None:
    """A config entry sets up the ai_task entity and marks it unavailable on unload."""
    entry = MockConfigEntry(
        domain=DOMAIN, unique_id=BASE_URL, data={CONF_BASE_URL: BASE_URL}
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    state = hass.states.get(ENTITY_ID)
    assert state is not None
    assert state.state != STATE_UNAVAILABLE

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
    # RestoreEntity keeps the entity in the state machine as "unavailable"
    # rather than removing it outright, so it can restore state on reload.
    state = hass.states.get(ENTITY_ID)
    assert state is not None
    assert state.state == STATE_UNAVAILABLE
