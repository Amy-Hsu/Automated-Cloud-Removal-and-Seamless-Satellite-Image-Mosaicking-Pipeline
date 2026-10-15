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
