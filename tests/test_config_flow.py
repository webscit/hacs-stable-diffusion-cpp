"""Tests for the Stable Diffusion cpp config and options flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.stable_diffusion_cpp.api import SDCppConnectionError
from custom_components.stable_diffusion_cpp.const import (
    CONF_BASE_URL,
    CONF_CFG_SCALE,
    CONF_HEIGHT,
    CONF_NEGATIVE_PROMPT,
    CONF_OUTPUT_COMPRESSION,
    CONF_OUTPUT_FORMAT,
    CONF_SAMPLER,
    CONF_SEED,
    CONF_STEPS,
    CONF_WIDTH,
    DOMAIN,
)

BASE_URL = "http://127.0.0.1:1234"


async def test_user_flow_success(hass: HomeAssistant) -> None:
    """A valid base URL that connects successfully creates a config entry."""
    with patch(
        "custom_components.stable_diffusion_cpp.config_flow.StableDiffusionCppClient.async_get_models",
        AsyncMock(return_value=["sd_v1.5"]),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_BASE_URL: BASE_URL}
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_BASE_URL: BASE_URL}


async def test_user_flow_cannot_connect(hass: HomeAssistant) -> None:
    """A connection failure re-shows the form with an error."""
    with patch(
        "custom_components.stable_diffusion_cpp.config_flow.StableDiffusionCppClient.async_get_models",
        AsyncMock(side_effect=SDCppConnectionError("boom")),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_BASE_URL: BASE_URL}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_duplicate_entry(hass: HomeAssistant) -> None:
    """A base URL matching an existing entry aborts as already configured."""
    entry = MockConfigEntry(
        domain=DOMAIN, unique_id=BASE_URL, data={CONF_BASE_URL: BASE_URL}
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.stable_diffusion_cpp.config_flow.StableDiffusionCppClient.async_get_models",
        AsyncMock(return_value=["sd_v1.5"]),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_BASE_URL: BASE_URL}
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_flow_round_trip(hass: HomeAssistant) -> None:
    """Submitting the options form stores the values verbatim."""
    entry = MockConfigEntry(
        domain=DOMAIN, unique_id=BASE_URL, data={CONF_BASE_URL: BASE_URL}
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    submitted = {
        CONF_NEGATIVE_PROMPT: "blurry",
        CONF_STEPS: 30,
        CONF_CFG_SCALE: 9.5,
        CONF_SAMPLER: "dpm++2m",
        CONF_SEED: 42,
        CONF_WIDTH: 768,
        CONF_HEIGHT: 512,
        CONF_OUTPUT_FORMAT: "jpeg",
        CONF_OUTPUT_COMPRESSION: 80,
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == submitted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (CONF_STEPS, 0),
        (CONF_CFG_SCALE, 999),
        (CONF_WIDTH, 16),
    ],
)
async def test_options_flow_validates_ranges(
    hass: HomeAssistant, field: str, value: int
) -> None:
    """Out-of-range values are rejected by the options form schema."""
    entry = MockConfigEntry(
        domain=DOMAIN, unique_id=BASE_URL, data={CONF_BASE_URL: BASE_URL}
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    base_values = {
        CONF_NEGATIVE_PROMPT: "",
        CONF_STEPS: 20,
        CONF_CFG_SCALE: 7.0,
        CONF_SAMPLER: "euler_a",
        CONF_SEED: 42,
        CONF_WIDTH: 512,
        CONF_HEIGHT: 512,
        CONF_OUTPUT_FORMAT: "png",
        CONF_OUTPUT_COMPRESSION: 0,
    }
    base_values[field] = value

    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"], base_values
        )
