# magazynflary audio

## Development

Install dependencies (including dev tools) with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Set up the git hooks (ruff, mypy, uv lock check):

```sh
pre-commit install
```

Run all hooks manually:

```sh
pre-commit run --all-files
```

Run tests:

```sh
uv run pytest
```
