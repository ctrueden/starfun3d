#!/bin/sh

if ! command -v pixi >/dev/null 2>&1; then
  echo "Please install pixi (https://pixi.sh/latest/#installation)."
  exit 1
fi
