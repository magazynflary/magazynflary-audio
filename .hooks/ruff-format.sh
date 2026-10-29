#!/bin/sh
exec uv run --locked ruff format "$@"
