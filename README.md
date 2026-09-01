# Stable Diffusion cpp for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)

A [Home Assistant](https://www.home-assistant.io/) custom integration that exposes a
self-hosted [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
server as an [`ai_task`](https://www.home-assistant.io/integrations/ai_task/)
`GENERATE_IMAGE` provider.

## Features

- Adds an `ai_task` entity that generates images via the `ai_task.generate_image`
  action, backed by your own stable-diffusion.cpp server.
- Talks to stable-diffusion.cpp's OpenAI-like `/v1/images/generations` endpoint.
- Default generation parameters (sampling steps, CFG scale, sampler, seed,
  negative prompt, image size, output format) are configured once via the
  integration's Options, since HA's `ai_task` action only ever passes a
  free-text prompt.

## Requirements

A running stable-diffusion.cpp server with its bundled HTTP server
(`examples/server`) started, reachable from Home Assistant.

## Installation

### HACS (recommended)

1. In HACS, go to **Integrations** → the **⋮** menu → **Custom repositories**.
2. Add `https://github.com/fcollonval/hacs-stable-diffusion-cpp` as an
   **Integration**.
3. Search for "Stable Diffusion cpp" in HACS and install it.
4. Restart Home Assistant.

### Manual

Copy `custom_components/stable_diffusion_cpp` into your Home Assistant
`custom_components` directory and restart Home Assistant.

## Configuration

1. Go to **Settings** → **Devices & services** → **Add integration**.
2. Search for "Stable Diffusion cpp".
3. Enter the base URL of your stable-diffusion.cpp server
   (e.g. `http://127.0.0.1:1234`).

### Options

After setup, click **Configure** on the integration to set the default
generation parameters used for every `ai_task.generate_image` call:

| Option               | Description                              | Default |
| -------------------- | ----------------------------------------- | ------- |
| Negative prompt       | Appended as a negative prompt             | (empty) |
| Sampling steps        | Number of sampling steps                  | 20      |
| CFG scale              | Classifier-free guidance scale            | 7.0     |
| Sampler                | Sampling method                           | euler_a |
| Seed                   | `-1` requests a fresh random seed on every call; any other value is fixed (deterministic) | -1 |
| Image width / height   | Requested output size                     | 512x512 |
| Output format          | `png`, `jpeg`, or `webp`                  | png     |
| Output compression     | Compression level (0-100)                 | 0       |

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```console
./scripts/setup    # install dependencies
./scripts/develop  # run a local Home Assistant instance with this integration loaded
./scripts/lint     # format and lint
uv run pytest      # run the test suite
```

## Contributing

Issues and pull requests are welcome.

## License

MIT — see [LICENSE](LICENSE).
