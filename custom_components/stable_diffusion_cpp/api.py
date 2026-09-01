"""API client for a stable-diffusion.cpp server's OpenAI-like image API."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import mimetypes
from dataclasses import dataclass
from typing import Any

import aiohttp

from .const import HTTP_OK, MIME_TYPES, MODELS_TIMEOUT, TIMEOUT


class SDCppApiError(Exception):
    """Base exception for the stable-diffusion.cpp API client."""


class SDCppConnectionError(SDCppApiError):
    """Raised on network/timeout errors reaching the server."""


class SDCppServerError(SDCppApiError):
    """Raised when the server returns a 4xx/5xx JSON error body."""

    def __init__(self, message: str, status: int) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.status = status


class SDCppResponseError(SDCppApiError):
    """Raised when the response body is malformed or missing expected fields."""


@dataclass(slots=True)
class GeneratedImage:
    """A single generated image."""

    image_data: bytes
    mime_type: str
    width: int | None
    height: int | None


class StableDiffusionCppClient:
    """Client for a stable-diffusion.cpp server's OpenAI-like image API."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession) -> None:
        """Initialize the client."""
        self._base_url = base_url.rstrip("/")
        self._session = session

    async def async_get_models(self) -> list[str]:
        """Return the list of model ids known to the server.

        Also used as the config-flow connectivity check.
        """
        try:
            async with asyncio.timeout(MODELS_TIMEOUT):
                resp = await self._session.get(f"{self._base_url}/v1/models")
                body = await resp.json()
                if resp.status != HTTP_OK:
                    raise SDCppServerError(
                        self._extract_error_message(body), resp.status
                    )
        except TimeoutError as err:
            raise SDCppConnectionError("Timeout connecting to server") from err
        except aiohttp.ClientError as err:
            raise SDCppConnectionError(str(err)) from err

        return [
            model["id"]
            for model in body.get("data", [])
            if isinstance(model, dict) and "id" in model
        ]

    async def async_generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str = "png",
        output_compression: int = 0,
        negative_prompt: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        seed: int | None = None,
    ) -> GeneratedImage:
        """Generate an image and return the decoded result."""
        full_prompt = self.build_prompt(
            prompt,
            self._collect_extra_args_options(
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                seed=seed,
            ),
        )
        payload = {
            "prompt": full_prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "output_format": output_format,
            "output_compression": output_compression,
        }

        try:
            async with asyncio.timeout(TIMEOUT):
                resp = await self._session.post(
                    f"{self._base_url}/v1/images/generations", json=payload
                )
                body = await resp.json()
                if resp.status != HTTP_OK:
                    raise SDCppServerError(
                        self._extract_error_message(body), resp.status
                    )
        except TimeoutError as err:
            raise SDCppConnectionError("Timeout generating image") from err
        except aiohttp.ClientError as err:
            raise SDCppConnectionError(str(err)) from err

        return self._parse_generation_response(body, output_format, width, height)

    async def async_edit_image(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],
        *,
        width: int,
        height: int,
        output_format: str = "png",
        output_compression: int = 0,
        negative_prompt: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        seed: int | None = None,
    ) -> GeneratedImage:
        """Edit one or more source images and return the decoded result.

        `images` is a list of (raw bytes, mime type) pairs, sent as repeated
        `image[]` multipart fields per sd.cpp's documented edits schema
        (examples/server/api.md) - the preferred field for one or more
        reference images; the legacy singular `image` field isn't needed.
        """
        if not images:
            # aiohttp.FormData only switches Content-Type to multipart/
            # form-data once a field carries a filename; with no images at
            # all it would silently send application/x-www-form-urlencoded
            # instead, which the server rejects with a confusing "Content-
            # Type must be multipart/form-data" - verified against a live
            # server - rather than a clear "no image" error.
            msg = "async_edit_image requires at least one image"
            raise ValueError(msg)

        full_prompt = self.build_prompt(
            prompt,
            self._collect_extra_args_options(
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                seed=seed,
            ),
        )

        form = aiohttp.FormData()
        form.add_field("prompt", full_prompt)
        form.add_field("n", "1")
        form.add_field("size", f"{width}x{height}")
        form.add_field("output_format", output_format)
        form.add_field("output_compression", str(output_compression))
        for image_data, mime_type in images:
            extension = mimetypes.guess_extension(mime_type) or ".png"
            form.add_field(
                "image[]",
                image_data,
                filename=f"image{extension}",
                content_type=mime_type,
            )

        try:
            async with asyncio.timeout(TIMEOUT):
                resp = await self._session.post(
                    f"{self._base_url}/v1/images/edits", data=form
                )
                body = await resp.json()
                if resp.status != HTTP_OK:
                    raise SDCppServerError(
                        self._extract_error_message(body), resp.status
                    )
        except TimeoutError as err:
            raise SDCppConnectionError("Timeout editing image") from err
        except aiohttp.ClientError as err:
            raise SDCppConnectionError(str(err)) from err

        return self._parse_generation_response(body, output_format, width, height)

    @staticmethod
    def _collect_extra_args_options(
        *,
        negative_prompt: str | None,
        steps: int | None,
        cfg_scale: float | None,
        sampler: str | None,
        seed: int | None,
    ) -> dict[str, Any]:
        """Collect the generation options actually set, for build_prompt."""
        options: dict[str, Any] = {}
        if negative_prompt:
            options["negative_prompt"] = negative_prompt
        if seed is not None:
            options["seed"] = seed
        if steps is not None:
            options["steps"] = steps
        if cfg_scale is not None:
            options["cfg_scale"] = cfg_scale
        if sampler is not None:
            options["sampler"] = sampler
        return options

    @staticmethod
    def build_prompt(prompt: str, options: dict[str, Any]) -> str:
        """Build the final prompt, embedding sd.cpp's extra-args block if needed.

        Verified schema (SDGenerationParams::from_json_str, examples/common/common.cpp):
        top-level "negative_prompt"/"seed", nested "sample_params.sample_steps",
        "sample_params.sample_method", "sample_params.guidance.txt_cfg". Only options
        actually present are embedded, and the wrapper is omitted entirely when there
        is nothing to override, so an unknown/misnamed key can only ever affect a
        customized entry, never a fresh install using server-side defaults.
        """
        extra_args: dict[str, Any] = {}
        if "negative_prompt" in options:
            extra_args["negative_prompt"] = options["negative_prompt"]
        if "seed" in options:
            extra_args["seed"] = options["seed"]

        sample_params: dict[str, Any] = {}
        if "steps" in options:
            sample_params["sample_steps"] = options["steps"]
        if "sampler" in options:
            sample_params["sample_method"] = options["sampler"]
        if "cfg_scale" in options:
            sample_params["guidance"] = {"txt_cfg": options["cfg_scale"]}
        if sample_params:
            extra_args["sample_params"] = sample_params

        if not extra_args:
            return prompt

        extra_args_json = json.dumps(extra_args)
        return f"<sd_cpp_extra_args>{extra_args_json}</sd_cpp_extra_args>{prompt}"

    @staticmethod
    def _extract_error_message(body: dict[str, Any]) -> str:
        """Extract the error message from a sd.cpp error body.

        sd.cpp returns either {"error": "msg"} (400) or
        {"error": "server_error", "message": "msg"} (500); message wins when present.
        """
        if "message" in body:
            return body["message"]
        return body.get("error", "Unknown error")

    @staticmethod
    def _parse_generation_response(
        body: dict[str, Any], requested_format: str, width: int, height: int
    ) -> GeneratedImage:
        """Parse a successful /v1/images/generations response."""
        try:
            b64_json = body["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError) as err:
            raise SDCppResponseError("Missing b64_json in response") from err

        try:
            image_data = base64.b64decode(b64_json, validate=True)
        except (binascii.Error, ValueError) as err:
            raise SDCppResponseError("Invalid base64 image data") from err

        output_format = body.get("output_format", requested_format)
        mime_type = MIME_TYPES.get(output_format, "image/png")

        return GeneratedImage(
            image_data=image_data, mime_type=mime_type, width=width, height=height
        )
