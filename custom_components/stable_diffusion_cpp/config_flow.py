"""Config flow for the Stable Diffusion cpp integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .api import SDCppApiError, StableDiffusionCppClient
from .const import (
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
    DEFAULT_BASE_URL,
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OUTPUT_COMPRESSION,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SAMPLER,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    DOMAIN,
    LOGGER,
    OUTPUT_FORMAT_OPTIONS,
    SAMPLER_OPTIONS,
)


class SDCppConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Stable Diffusion cpp."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].rstrip("/")
            session = aiohttp_client.async_get_clientsession(self.hass)
            client = StableDiffusionCppClient(base_url, session)
            try:
                await client.async_get_models()
            except SDCppApiError:
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                await self.async_set_unique_id(base_url)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=base_url, data={CONF_BASE_URL: base_url}
                )

        schema = vol.Schema(
            {vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str}
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SDCppOptionsFlow:
        """Create the options flow."""
        return SDCppOptionsFlow()


class SDCppOptionsFlow(OptionsFlow):
    """Handle an options flow for Stable Diffusion cpp."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the generation-parameter options."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_NEGATIVE_PROMPT,
                    default=options.get(CONF_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT),
                ): str,
                vol.Optional(
                    CONF_STEPS, default=options.get(CONF_STEPS, DEFAULT_STEPS)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=150)),
                vol.Optional(
                    CONF_CFG_SCALE,
                    default=options.get(CONF_CFG_SCALE, DEFAULT_CFG_SCALE),
                ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=30.0)),
                vol.Optional(
                    CONF_SAMPLER, default=options.get(CONF_SAMPLER, DEFAULT_SAMPLER)
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=SAMPLER_OPTIONS, mode=SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Optional(
                    CONF_SEED, default=options.get(CONF_SEED, DEFAULT_SEED)
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_WIDTH, default=options.get(CONF_WIDTH, DEFAULT_WIDTH)
                ): vol.All(vol.Coerce(int), vol.Range(min=64, max=2048)),
                vol.Optional(
                    CONF_HEIGHT, default=options.get(CONF_HEIGHT, DEFAULT_HEIGHT)
                ): vol.All(vol.Coerce(int), vol.Range(min=64, max=2048)),
                vol.Optional(
                    CONF_OUTPUT_FORMAT,
                    default=options.get(CONF_OUTPUT_FORMAT, DEFAULT_OUTPUT_FORMAT),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=OUTPUT_FORMAT_OPTIONS,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_OUTPUT_COMPRESSION,
                    default=options.get(
                        CONF_OUTPUT_COMPRESSION, DEFAULT_OUTPUT_COMPRESSION
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
