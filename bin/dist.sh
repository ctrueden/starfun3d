#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

pixi run -e dev python -m build
