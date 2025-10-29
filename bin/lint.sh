#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0

# Check for errors and capture non-zero exit codes.
pixi run -e dev validate-pyproject pyproject.toml
code=$?; test $code -eq 0 || exitCode=$code
pixi run -e dev ruff check >/dev/null 2>&1
code=$?; test $code -eq 0 || exitCode=$code
pixi run -e dev ruff format --check >/dev/null 2>&1
code=$?; test $code -eq 0 || exitCode=$code

# Do actual code reformatting.
pixi run -e dev ruff check --fix
code=$?; test $code -eq 0 || exitCode=$code
pixi run -e dev ruff format
code=$?; test $code -eq 0 || exitCode=$code

exit $exitCode
