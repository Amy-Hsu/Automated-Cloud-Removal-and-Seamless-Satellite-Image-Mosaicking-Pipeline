"""Load the pipeline step scripts as importable modules for testing."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def step2():
    return _load("step2mosaic", "step2_mosaic/step2mosaic.py")


@pytest.fixture(scope="session")
def step4():
    return _load("step4cloud_removal", "step4_cloud_removal/step4cloud_removal.py")
