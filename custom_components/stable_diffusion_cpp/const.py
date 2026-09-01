"""Constants for the Stable Diffusion cpp integration."""

import logging

DOMAIN = "stable_diffusion_cpp"
LOGGER = logging.getLogger(__package__)

# Config entry keys (connection, set once at setup, not editable via options)
CONF_BASE_URL = "base_url"

# Options keys (generation defaults, editable via the Options flow)
CONF_NEGATIVE_PROMPT = "negative_prompt"
CONF_STEPS = "steps"
CONF_CFG_SCALE = "cfg_scale"
CONF_SAMPLER = "sampler"
CONF_SEED = "seed"
CONF_WIDTH = "width"
CONF_HEIGHT = "height"
CONF_OUTPUT_FORMAT = "output_format"
CONF_OUTPUT_COMPRESSION = "output_compression"

# Defaults (mirror stable-diffusion.cpp server defaults where one exists)
DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 7.0
DEFAULT_SAMPLER = "euler_a"
# -1 tells the server to pick a fresh random seed every call (verified in
# SDGenerationParams::resolve(), examples/common/common.cpp: `if (seed < 0)
# { srand(...); seed = rand(); }`). Omitting the field entirely is NOT the
# same thing: the struct's own compiled-in default is a literal, fixed 42,
# which would make every generation for a given prompt identical. Unlike the
# other generation options, seed is therefore always sent explicitly.
DEFAULT_SEED = -1
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_OUTPUT_COMPRESSION = 0

# Verified against sample_method_to_str[] in stable-diffusion.cpp's
# src/stable-diffusion.cpp; unknown strings are silently ignored server-side.
SAMPLER_OPTIONS = [
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
    "lms",
]
OUTPUT_FORMAT_OPTIONS = ["png", "jpeg", "webp"]

TIMEOUT = 120  # seconds; image generation can be slow on CPU-bound servers
MODELS_TIMEOUT = 10  # seconds; used for the config-flow connectivity check

HTTP_OK = 200

MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
