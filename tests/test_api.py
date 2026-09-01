"""Tests for the stable-diffusion.cpp API client."""

from __future__ import annotations

import asyncio
import base64
import json

import aiohttp
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.stable_diffusion_cpp.api import (
    SDCppConnectionError,
    SDCppResponseError,
    SDCppServerError,
    StableDiffusionCppClient,
)
from custom_components.stable_diffusion_cpp.const import DEFAULT_SEED

BASE_URL = "http://127.0.0.1:1234"
FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"
FAKE_B64 = base64.b64encode(FAKE_PNG_BYTES).decode()


@pytest.fixture
async def client(
    aioclient_mock: AiohttpClientMocker,
) -> StableDiffusionCppClient:
    """Return a client bound to a session backed by HA's aioclient_mock."""
    session = aioclient_mock.create_session(asyncio.get_running_loop())
    try:
        yield StableDiffusionCppClient(BASE_URL, session)
    finally:
        await session.close()


def test_build_prompt_no_wrapper_when_options_empty() -> None:
    """No sd_cpp_extra_args wrapper is added when nothing is customized."""
    prompt = StableDiffusionCppClient.build_prompt("a red bicycle", {})
    assert prompt == "a red bicycle"
    assert "sd_cpp_extra_args" not in prompt


def test_build_prompt_embeds_verified_nested_schema() -> None:
    """Extra args use the verified nested sample_params/guidance schema."""
    prompt = StableDiffusionCppClient.build_prompt(
        "a red bicycle",
        {
            "negative_prompt": "blurry",
            "seed": 42,
            "steps": 30,
            "cfg_scale": 9.5,
            "sampler": "dpm++2m",
        },
    )
    assert prompt.startswith("<sd_cpp_extra_args>")
    assert prompt.endswith("</sd_cpp_extra_args>a red bicycle")

    raw_json = prompt[len("<sd_cpp_extra_args>") : prompt.index("</sd_cpp_extra_args>")]
    extra_args = json.loads(raw_json)

    assert extra_args["negative_prompt"] == "blurry"
    assert extra_args["seed"] == 42
    assert extra_args["sample_params"]["sample_steps"] == 30
    assert extra_args["sample_params"]["sample_method"] == "dpm++2m"
    assert extra_args["sample_params"]["guidance"]["txt_cfg"] == 9.5

    # No flat keys, and no width/height (those travel via the top-level "size" field).
    assert "steps" not in extra_args
    assert "cfg_scale" not in extra_args
    assert "sampler" not in extra_args
    assert "width" not in extra_args
    assert "height" not in extra_args


def test_build_prompt_only_includes_set_options() -> None:
    """Only options actually present are embedded; partial dicts are supported."""
    prompt = StableDiffusionCppClient.build_prompt("cat", {"seed": 7})
    raw_json = prompt[len("<sd_cpp_extra_args>") : prompt.index("</sd_cpp_extra_args>")]
    extra_args = json.loads(raw_json)
    assert extra_args == {"seed": 7}


async def test_generate_image_success(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A successful generation returns decoded image bytes and mime type."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        json={
            "created": 1234567890,
            "output_format": "png",
            "data": [{"b64_json": FAKE_B64}],
        },
    )
    image = await client.async_generate_image(
        "a cat", width=512, height=512, output_format="png"
    )

    assert image.image_data == FAKE_PNG_BYTES
    assert image.mime_type == "image/png"
    assert image.width == 512
    assert image.height == 512


async def test_generate_image_size_string_format(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """The request's size field is WIDTHxHEIGHT, joined at the transport boundary."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        json={"data": [{"b64_json": FAKE_B64}]},
    )
    await client.async_generate_image("a cat", width=768, height=512)

    assert aioclient_mock.call_count == 1
    _, _, data, _ = aioclient_mock.mock_calls[0]
    assert data["size"] == "768x512"


async def test_generate_image_default_seed_only_wraps_prompt(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """The exact wire format the ai_task entity sends on a fresh install.

    Pins the composition of entity defaults + client serialization: seed is
    always sent (DEFAULT_SEED = -1, requesting server-side randomization),
    and nothing else is wrapped when no other option is set.
    """
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        json={"data": [{"b64_json": FAKE_B64}]},
    )
    await client.async_generate_image("a cat", width=512, height=512, seed=DEFAULT_SEED)

    _, _, data, _ = aioclient_mock.mock_calls[0]
    assert data["prompt"] == '<sd_cpp_extra_args>{"seed": -1}</sd_cpp_extra_args>a cat'


async def test_generate_image_400_error(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A 400 response with a flat error body raises SDCppServerError."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        status=400,
        json={"error": "invalid sd_cpp_extra_args"},
    )
    with pytest.raises(SDCppServerError) as exc_info:
        await client.async_generate_image("a cat", width=512, height=512)

    assert exc_info.value.status == 400
    assert str(exc_info.value) == "invalid sd_cpp_extra_args"


async def test_generate_image_500_error(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A 500 response prefers the 'message' field over 'error'."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        status=500,
        json={"error": "server_error", "message": "boom"},
    )
    with pytest.raises(SDCppServerError) as exc_info:
        await client.async_generate_image("a cat", width=512, height=512)

    assert exc_info.value.status == 500
    assert str(exc_info.value) == "boom"


async def test_generate_image_malformed_response(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A 200 response missing the data field raises SDCppResponseError."""
    aioclient_mock.post(f"{BASE_URL}/v1/images/generations", json={"created": 1})
    with pytest.raises(SDCppResponseError):
        await client.async_generate_image("a cat", width=512, height=512)


async def test_generate_image_invalid_base64(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """Invalid base64 image data raises SDCppResponseError."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        json={"data": [{"b64_json": "not-valid-base64!!!"}]},
    )
    with pytest.raises(SDCppResponseError):
        await client.async_generate_image("a cat", width=512, height=512)


async def test_generate_image_connection_error(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A connection failure raises SDCppConnectionError."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations",
        exc=aiohttp.ClientConnectionError("boom"),
    )
    with pytest.raises(SDCppConnectionError):
        await client.async_generate_image("a cat", width=512, height=512)


async def test_generate_image_timeout(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A timeout raises SDCppConnectionError."""
    aioclient_mock.post(
        f"{BASE_URL}/v1/images/generations", exc=TimeoutError("timed out")
    )
    with pytest.raises(SDCppConnectionError):
        await client.async_generate_image("a cat", width=512, height=512)


async def test_get_models_success(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """GET /v1/models returns a list of model ids."""
    aioclient_mock.get(
        f"{BASE_URL}/v1/models",
        json={"data": [{"id": "sd_v1.5"}, {"id": "sd_xl"}]},
    )
    models = await client.async_get_models()

    assert models == ["sd_v1.5", "sd_xl"]


async def test_get_models_connection_error(
    client: StableDiffusionCppClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """A connection failure while listing models raises SDCppConnectionError."""
    aioclient_mock.get(
        f"{BASE_URL}/v1/models", exc=aiohttp.ClientConnectionError("boom")
    )
    with pytest.raises(SDCppConnectionError):
        await client.async_get_models()
