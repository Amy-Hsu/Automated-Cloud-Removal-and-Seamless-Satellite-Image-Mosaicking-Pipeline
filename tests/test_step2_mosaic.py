"""Unit tests for the seamline and blending helpers in Step 2."""
import numpy as np


def test_add_at_pos_inside(step2):
    canvas = np.zeros((5, 5))
    step2.add_at_pos(canvas, np.ones((2, 3)), (1, 2))
    assert canvas.sum() == 6
    assert canvas[2:4, 1:4].all()


def test_add_at_pos_clips_off_edges(step2):
    canvas = np.zeros((4, 4))
    out = step2.add_at_pos(canvas, np.ones((3, 3)), (-1, 2), inPlace=False)
    assert canvas.sum() == 0  # original untouched
    assert out.sum() == 4     # only the 2x2 overlap is added
    assert out[2:4, 0:2].all()


def test_spiral_starts_at_origin_and_is_unique(step2):
    pts = step2.spiral(5, 5)
    assert pts[0] == [0, 0]
    assert len(pts) == 25
    assert len({tuple(p) for p in pts}) == 25
    # neighbours of the origin come before anything farther away
    assert {tuple(p) for p in pts[1:9]} == {
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
    }


def test_bfs_finds_shortest_path_around_wall(step2):
    grid = [
        [0, 0, 0, 0],
        [9, 9, 9, 0],
        [2, 0, 0, 0],
    ]
    path = step2.bfs(grid, (0, 0), wall=9, clear=0, goal=2, width=4, height=3)
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)
    assert len(path) == 9  # 3 right, 2 down, 3 left
    assert all(grid[y][x] != 9 for x, y in path)


def test_bfs_returns_none_when_unreachable(step2):
    grid = [[0, 9, 2]]
    assert step2.bfs(grid, (0, 0), 9, 0, 2, 3, 1) is None


def test_blend_mask_is_feathered(step2):
    mask = np.zeros((41, 41), dtype=np.uint8)
    mask[:, 20:] = 1
    blended = step2.blendMask(11, mask)
    assert blended.dtype == np.float32
    assert blended.min() >= 0 and blended.max() <= 1
    row = blended[20]
    assert row[0] == 0 and row[-1] == 1
    assert 0 < row[20] < 1          # soft transition across the seam
    assert np.all(np.diff(row) >= 0)


def test_build_graph_seamline_follows_low_cost_edge(step2):
    """The seamline should run along the column where the two images differ least.

    Edge weights are inverse image gradients, so the cheapest path lies along
    strong gradients in both images; here a vertical edge at column 4.
    """
    h, w = 9, 9
    img = np.full((h, w), 10, dtype=np.uint8)
    img[:, 5:] = 200
    source = 0 * w + 4 + 1          # node ids are 1-based, row-major
    target = (h - 2) * w + 4 + 1
    seam = step2.buildGraph(img, img.copy(), h, w, "a7c", (source, target))
    assert seam.shape == (h, w)
    rows, cols = np.nonzero(seam)
    assert seam[0, 4] == 1 and seam[h - 2, 4] == 1
    assert set(cols) <= {4, 5}
    assert set(range(h - 1)) <= set(rows)


def test_build_graph_links_every_pixel_to_its_right_and_down_neighbours(step2, monkeypatch):
    """Every valid pixel gets an edge to its right and lower neighbour.

    Node ids are 1-based and row-major with ``width`` columns, so only the last
    column (x % width == 0) and the last row lack a right or down neighbour.
    A height that does not divide the width checks that no other nodes are
    skipped (issue #3: an extra ``x % height != 0`` condition dropped them).
    """
    graphs = []
    astar_path = step2.nx.astar_path

    def spy(G, *args, **kwargs):
        graphs.append(G)
        return astar_path(G, *args, **kwargs)

    monkeypatch.setattr(step2.nx, "astar_path", spy)
    h, w = 3, 4
    img = np.arange(1, h * w + 1, dtype=np.uint8).reshape(h, w)
    step2.buildGraph(img, img.copy(), h, w, "a7", (1, h * w))

    expected = set()
    for x in range(1, h * w + 1):
        if x % w != 0:
            expected.add(frozenset((x, x + 1)))
        if x <= (h - 1) * w:
            expected.add(frozenset((x, x + w)))
    assert {frozenset(e) for e in graphs[0].edges()} == expected


def test_polygonize_valid_area_outlines_footprint(step2, tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin

    data = np.zeros((2, 50, 60), np.uint16)
    data[:, 10:40, 5:45] = 500
    tif = tmp_path / "scene.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=60, height=50, count=2, dtype="uint16",
                       crs="EPSG:3826", transform=from_origin(1000, 2000, 10, 10)) as dst:
        dst.write(data)
    step2.polygonize_valid_area(str(tif))
    shp = gpd.read_file(tmp_path / "scene.temp.shp")
    assert len(shp) == 1
    assert shp.area.iloc[0] == 30 * 40 * 10 * 10
    assert shp.total_bounds.tolist() == [1050, 1600, 1450, 1900]
