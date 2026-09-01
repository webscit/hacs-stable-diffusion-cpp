"""Tests for the Stable Diffusion cpp ai_task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import ai_task
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.stable_diffusion_cpp.ai_task import SDCppTaskEntity
from custom_components.stable_diffusion_cpp.api import (
    GeneratedImage,
    SDCppConnectionError,
)
from custom_components.stable_diffusion_cpp.const import (
    CONF_CFG_SCALE,
    CONF_HEIGHT,
    CONF_NEGATIVE_PROMPT,
    CONF_OUTPUT_COMPRESSION,
    CONF_OUTPUT_FORMAT,
    CONF_SAMPLER,
    CONF_SEED,
    CONF_STEPS,
    CONF_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_OUTPUT_COMPRESSION,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
)
from custom_components.stable_diffusion_cpp.data import SDCppData


def _make_entry(options: dict | None = None) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain="stable_diffusion_cpp",
        data={"base_url": "http://127.0.0.1:1234"},
        options=options or {},
    )
    entry.runtime_data = SDCppData(client=AsyncMock())
    return entry


def _make_chat_log() -> MagicMock:
    chat_log = MagicMock()
    chat_log.conversation_id = "abc123"
    return chat_log


def test_supported_features() -> None:
    """Only GENERATE_IMAGE is supported."""
    entity = SDCppTaskEntity(_make_entry())
    assert entity.supported_features == ai_task.AITaskEntityFeature.GENERATE_IMAGE


async def test_generate_image_happy_path() -> None:
    """A successful generation is mapped to a GenImageTaskResult."""
    entry = _make_entry()
    entry.runtime_data.client.async_generate_image = AsyncMock(
        return_value=GeneratedImage(
            image_data=b"pngdata", mime_type="image/png", width=512, height=512
        )
    )
    entity = SDCppTaskEntity(entry)
    chat_log = _make_chat_log()

    result = await entity._async_generate_image(
        ai_task.GenImageTask(name="task", instructions="a cat"), chat_log
    )

    assert isinstance(result, ai_task.GenImageTaskResult)
    assert result.image_data == b"pngdata"
    assert result.mime_type == "image/png"
    assert result.width == 512
    assert result.height == 512
    assert result.conversation_id == "abc123"


async def test_generate_image_passes_options() -> None:
    """Options stored on the config entry are threaded into the client call."""
    entry = _make_entry(
        {
            CONF_NEGATIVE_PROMPT: "blurry",
            CONF_STEPS: 30,
            CONF_CFG_SCALE: 9.5,
            CONF_SAMPLER: "dpm++2m",
            CONF_SEED: 7,
            CONF_WIDTH: 768,
            CONF_HEIGHT: 512,
            CONF_OUTPUT_FORMAT: "jpeg",
            CONF_OUTPUT_COMPRESSION: 80,
        }
    )
    entry.runtime_data.client.async_generate_image = AsyncMock(
        return_value=GeneratedImage(
            image_data=b"x", mime_type="image/jpeg", width=768, height=512
        )
    )
    entity = SDCppTaskEntity(entry)

    await entity._async_generate_image(
        ai_task.GenImageTask(name="task", instructions="a cat"), _make_chat_log()
    )

    entry.runtime_data.client.async_generate_image.assert_awaited_once_with(
        "a cat",
        negative_prompt="blurry",
        steps=30,
        cfg_scale=9.5,
        sampler="dpm++2m",
        seed=7,
        width=768,
        height=512,
        output_format="jpeg",
        output_compression=80,
    )


async def test_generate_image_uses_defaults_when_options_empty() -> None:
    """Only seed/size/format are sent when options are empty.

    Seed is always sent (to request server-side randomization), so the
    sd_cpp_extra_args wrapper is still emitted on a fresh install, but with
    only {"seed": -1} in it; negative_prompt/steps/cfg_scale/sampler are
    omitted entirely rather than sent as client-side defaults (see
    test_api.py::test_generate_image_default_seed_only_wraps_prompt for the
    exact wire format this produces).
    """
    entry = _make_entry()
    entry.runtime_data.client.async_generate_image = AsyncMock(
        return_value=GeneratedImage(
            image_data=b"x", mime_type="image/png", width=512, height=512
        )
    )
    entity = SDCppTaskEntity(entry)

    await entity._async_generate_image(
        ai_task.GenImageTask(name="task", instructions="a cat"), _make_chat_log()
    )

    entry.runtime_data.client.async_generate_image.assert_awaited_once_with(
        "a cat",
        seed=DEFAULT_SEED,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        output_format=DEFAULT_OUTPUT_FORMAT,
        output_compression=DEFAULT_OUTPUT_COMPRESSION,
    )


async def test_generate_image_error_propagation() -> None:
    """API errors are converted to HomeAssistantError, not leaked raw."""
    entry = _make_entry()
    entry.runtime_data.client.async_generate_image = AsyncMock(
        side_effect=SDCppConnectionError("down")
    )
    entity = SDCppTaskEntity(entry)

    with pytest.raises(HomeAssistantError):
        await entity._async_generate_image(
            ai_task.GenImageTask(name="task", instructions="a cat"), _make_chat_log()
        )
