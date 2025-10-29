# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarFun3D is a simple 3D nuclei segmentation playground using pre-trained StarDist models. The project generates synthetic nuclei volumes and segments them using deep learning models trained on fluorescence microscopy data.

## Key Commands

**Prerequisites**: Requires `uv` package manager (https://docs.astral.sh/uv/getting-started/installation/)

**Running the application**:
```bash
uv run src/main.py
```

**Linting and formatting**:
```bash
make lint
# Or directly:
uv run ruff check --fix
uv run ruff format
```

**Running tests**:
```bash
make test
# Or with specific test:
uv run python -m pytest -v -p no:faulthandler tests
uv run python -m pytest -v -p no:faulthandler test_file.py::test_function
```

**Validation**:
```bash
uv run validate-pyproject pyproject.toml
```

**Building distribution**:
```bash
make dist
```

## Architecture

### Core Modules

**`src/segment.py`**: 3D segmentation engine
- `segment_3d()`: Main segmentation function using StarDist3D models
- `segment_3d_batch()`: Batch processing for multiple volumes
- Uses CSBDeep for normalization and StarDist for object detection
- Model path defaults to `models/confocal` but supports three models (see Models section)

**`src/main.py`**: Demo/visualization script
- `make_synthetic_nuclei()`: Generates synthetic 3D fluorescence volumes with Gaussian blob nuclei
- Creates 64x256x256 volumes by default
- Uses `ndv` (n-dimensional viewer) with Qt backend for 3D visualization
- Stacks original volume and segmentation labels as separate channels

### Models

Located in `models/` directory. Three pre-trained StarDist3D models available:
- `confocal`: For confocal acquisitions (FUCCI label), avg nucleus size [39, 39, 7] pixels
- `sospim`: For SOSPIM fluorescent images (DAPI/SOX2), avg nucleus size [27, 28, 10] pixels
- `spinning`: For spinning-disk acquisitions (DAPI), avg nucleus size [39, 39, 7] pixels

Models from: Galindo et al. (2023), DOI: 10.1101/2023.12.06.570366

### Dependencies

- Python 3.10+
- Core ML: `stardist`, `tensorflow`, `csbdeep`, `numpy<2`, `scipy`
- Visualization: `ndv[qt]`, `pygfx<0.13.0`
- Dev tools: `pytest`, `ruff`, `validate-pyproject[all]`

### Project Structure

- Source code in `src/` (uses `src` layout with `package-dir = {"" = "src"}`)
- Models in `models/` (pre-trained weights and configs)
- Build scripts in `bin/` (shell scripts for common tasks)
- Dev tools configured via `pyproject.toml` (ruff, pytest settings)
- Academic paper in `paper/` folder for reference as needed
