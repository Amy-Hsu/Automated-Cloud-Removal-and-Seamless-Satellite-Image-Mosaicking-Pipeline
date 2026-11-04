"""Opt-in integer costs fix uint8 arithmetic while preserving historical methods."""
import warnings
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def graph_run(step2, monkeypatch):
    graphs = []

    def route(graph, *args, **kwargs):
        graphs.append(graph)
        return nx.astar_path(graph, *args, **kwargs)

    # Observe the graph while retaining NetworkX's actual shortest-path search.
    monkeypatch.setattr(step2, "nx", SimpleNamespace(Graph=nx.Graph, astar_path=route))

    def run(first, second, method):
        height, width = first.shape
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            seam = step2.buildGraph(first, second, height, width, method, (1, height * width))
        return graphs[-1], seam, caught

    return run


@pytest.mark.parametrize("method", ["a6c_int", "a7c_int"])
@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("values", [(10, 200), (200, 10), (50, 50)])
def test_integer_costs_have_signed_differences_and_the_original_zero_penalty(graph_run, method, vertical, values):
    image = np.array([values, values], np.uint8)
    if vertical:
        image = image.T
    graph, _, caught = graph_run(image, image.copy(), method)
    edge = (1, 3) if vertical else (1, 2)
    if values[0] == values[1]:
        expected = 2e10 if method == "a6c_int" else 1e10
    else:
        expected = 2 / 190 if method == "a6c_int" else 1 / 380
    assert graph.edges[edge]["weight"] == pytest.approx(expected, rel=1e-14)
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


@pytest.mark.parametrize("method,left,right,expected", [
    ("a6c", 10, 200, 2 / 66), ("a6c", 200, 10, 2 / 190),
    ("a7c", 10, 200, 1 / 132), ("a7c", 200, 10, 1 / 124),
    ("a6c", 50, 50, 2e10), ("a7c", 50, 50, 1e10),
])
def test_legacy_uint8_costs_remain_reproducible(graph_run, method, left, right, expected):
    image = np.array([[left, right], [left, right]], np.uint8)
    graph, _, _ = graph_run(image, image.copy(), method)
    assert graph.edges[1, 2]["weight"] == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("method", ["a6c_int", "a7c_int"])
@pytest.mark.parametrize("seed", [0, 1, 2, 17, 42])
def test_integer_costs_match_float_reference_and_intensity_inversion(graph_run, method, seed):
    rng = np.random.default_rng(seed)
    first = rng.integers(1, 255, (5, 5), dtype=np.uint8)
    second = rng.integers(1, 255, (5, 5), dtype=np.uint8)
    original = first.copy(), second.copy()
    integer_graph, seam, caught = graph_run(first, second, method)
    reference_graph, reference, _ = graph_run(first.astype(float), second.astype(float), method[:-4])
    inverted_graph, inverted, _ = graph_run(255 - first, 255 - second, method)
    for left, right, edge in integer_graph.edges(data=True):
        assert edge["weight"] == reference_graph.edges[left, right]["weight"]
        assert edge["weight"] == inverted_graph.edges[left, right]["weight"]
    np.testing.assert_array_equal(seam, reference)
    np.testing.assert_array_equal(seam, inverted)
    np.testing.assert_array_equal(first, original[0])
    np.testing.assert_array_equal(second, original[1])
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
