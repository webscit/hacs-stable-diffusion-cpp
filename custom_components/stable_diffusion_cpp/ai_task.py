"""AI Task platform for the Stable Diffusion cpp integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import SDCppApiError
from .const import (
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
from .data import SDCppConfigEntry
from .entity import SDCppEntity

# Options that should only be sent when the user actually set them via the
# Options flow. Seed is always sent (see below), so an uncustomized entry's
# sd_cpp_extra_args wrapper still contains {"seed": -1} - but a key-name
# mistake in any of *these* nested sample_params keys can only ever reach a
# customized entry, never a fresh install (see api.build_prompt).
_PASSTHROUGH_OPTIONS = (
    (CONF_NEGATIVE_PROMPT, "negative_prompt"),
    (CONF_STEPS, "steps"),
    (CONF_CFG_SCALE, "cfg_scale"),
    (CONF_SAMPLER, "sampler"),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SDCppConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the ai_task entity."""
    async_add_entities([SDCppTaskEntity(entry)])


class SDCppTaskEntity(SDCppEntity, ai_task.AITaskEntity):
    """AI Task entity backed by a stable-diffusion.cpp server."""

    _attr_name = "Image generation"
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_IMAGE

    def __init__(self, entry: SDCppConfigEntry) -> None:
        """Initialize the entity."""
        super().__init__(entry.entry_id, "ai_task")
        self._entry = entry

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        client = self._entry.runtime_data.client
        options = self._entry.options

        kwargs: dict[str, Any] = {
            # Always sent: -1 requests a fresh random seed each call, unlike
            # the other generation options which are omitted entirely when
            # unset (see _PASSTHROUGH_OPTIONS above).
            "seed": options.get(CONF_SEED, DEFAULT_SEED),
            "width": options.get(CONF_WIDTH, DEFAULT_WIDTH),
            "height": options.get(CONF_HEIGHT, DEFAULT_HEIGHT),
            "output_format": options.get(CONF_OUTPUT_FORMAT, DEFAULT_OUTPUT_FORMAT),
            "output_compression": options.get(
                CONF_OUTPUT_COMPRESSION, DEFAULT_OUTPUT_COMPRESSION
            ),
        }
        for conf_key, kwarg_name in _PASSTHROUGH_OPTIONS:
            if conf_key in options:
                kwargs[kwarg_name] = options[conf_key]

        try:
            image = await client.async_generate_image(task.instructions, **kwargs)
        except SDCppApiError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return ai_task.GenImageTaskResult(
            image_data=image.image_data,
            conversation_id=chat_log.conversation_id,
            mime_type=image.mime_type,
            width=image.width,
            height=image.height,
        )
