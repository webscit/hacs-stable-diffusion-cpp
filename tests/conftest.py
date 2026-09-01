"""Fixtures for the Stable Diffusion cpp integration tests."""

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.stable_diffusion_cpp.const import CONF_BASE_URL, DOMAIN


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for every test."""


@pytest.fixture(autouse=True)
async def auto_setup_homeassistant_component(hass: HomeAssistant) -> None:
    """Set up the core "homeassistant" component.

    Our manifest depends on ai_task -> conversation, whose default agent
    calls homeassistant.components.homeassistant.exposed_entities on
    EVENT_HOMEASSISTANT_STARTED. That data is only populated by the core
    "homeassistant" component's own async_setup, which isn't loaded by the
    bare `hass` fixture.
    """
    await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return a mock config entry with default options."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="http://127.0.0.1:1234",
        data={CONF_BASE_URL: "http://127.0.0.1:1234"},
        options={},
        unique_id="http://127.0.0.1:1234",
    )
