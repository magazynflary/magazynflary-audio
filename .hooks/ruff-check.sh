#!/bin/sh
exec uv run --locked ruff check --fix "$@"
