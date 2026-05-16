"""Smoke tests for the freud-ported analyzers exposed via PyO3.

Each test exercises the Python wrapper end-to-end: construct, build a
NeighborList where required, call compute, and sanity-check the shape
and a known value. Numerical correctness is covered by the Rust unit
tests in molrs-compute — these are wiring-level checks.
"""
import numpy as np
import pytest
import molrs


def _make_frame(pts, box_len=10.0):
    """Wrap an (N, 3) ndarray into a Frame with simbox."""
    f = molrs.Frame()
    b = molrs.Block()
    b.insert("x", np.ascontiguousarray(pts[:, 0], dtype=np.float64))
    b.insert("y", np.ascontiguousarray(pts[:, 1], dtype=np.float64))
    b.insert("z", np.ascontiguousarray(pts[:, 2], dtype=np.float64))
    f["atoms"] = b
    f.simbox = molrs.Box.cube(box_len)
    return f


def _octahedron_frame(cx=5.0, cy=5.0, cz=5.0, box_len=20.0):
    """Centre particle + ±x/y/z neighbors at unit distance."""
    pts = np.array(
        [
            [cx, cy, cz],
            [cx + 1, cy, cz],
            [cx - 1, cy, cz],
            [cx, cy + 1, cz],
            [cx, cy - 1, cz],
            [cx, cy, cz + 1],
            [cx, cy, cz - 1],
        ],
        dtype=np.float64,
    )
    return _make_frame(pts, box_len=box_len), pts


def _nlist(frame, pts, cutoff=1.2):
    nq = molrs.NeighborQuery(frame.simbox, pts, cutoff)
    return nq.query_self()


class TestSteinhardt:
    def test_ql_finite_on_octahedron(self):
        frame, pts = _octahedron_frame()
        nl = _nlist(frame, pts)
        s = molrs.Steinhardt(l=[6])
        out = s.compute(frame, nl)
        assert len(out) == 1
        ql = out[0]["ql"][0]
        # Centre particle has nonzero q_6; outer particles have 1 neighbor each.
        assert ql[0] > 0.0
        assert ql.shape == (7,)


class TestNematic:
    def test_aligned_gives_unity(self):
        frame, _ = _octahedron_frame()
        order, eigs, director, q = molrs.Nematic().compute(
            frame, [[0.0, 0.0, 1.0]] * 5
        )
        assert abs(order - 1.0) < 1e-10
        assert eigs.shape == (3,)
        assert q.shape == (3, 3)


class TestHexatic:
    def test_psi_shape(self):
        frame, pts = _octahedron_frame()
        nl = _nlist(frame, pts)
        out = molrs.Hexatic(k=6).compute(frame, nl)
        assert len(out) == 1
        psi = out[0]
        assert psi.shape == (7, 2)  # real / imag pairs


class TestSolidLiquid:
    def test_returns_arrays(self):
        frame, pts = _octahedron_frame()
        nl = _nlist(frame, pts)
        out = molrs.SolidLiquid(l=6, q_threshold=-2.0, n_threshold=1).compute(frame, nl)
        n_solid, is_solid = out[0]
        # All bonds count as "solid" with threshold = -2.
        assert n_solid[0] == 6
        assert len(is_solid) == 7


class TestLocalDensity:
    def test_pair_count(self):
        frame, pts = _octahedron_frame()
        nl = _nlist(frame, pts, cutoff=2.0)
        num, density = molrs.LocalDensity(r_max=2.0).compute(frame, nl)[0]
        # Centre particle has 6 neighbors within r=2.
        assert abs(num[0] - 6.0) < 1e-9
        assert density.shape == (7,)


class TestGaussianDensity:
    def test_integral(self):
        frame, _ = _octahedron_frame(box_len=10.0)
        grids = molrs.GaussianDensity(40, 40, 40, sigma=0.4).compute(frame)
        g = grids[0]
        assert g.shape == (40, 40, 40)
        voxel = (10.0 / 40) ** 3
        # Integral ≈ N (7 particles), within Gaussian-truncation error.
        assert abs(g.sum() * voxel - 7.0) < 0.5


class TestBondOrder:
    def test_octahedron_counts(self):
        frame, pts = _octahedron_frame()
        nl = _nlist(frame, pts)
        out = molrs.BondOrder(8, 8).compute(frame, nl)
        counts, bo, t_edges, p_edges = out[0]
        # 6 unique self-query bonds × 2 (symmetric counterparts) = 12.
        assert counts.sum() == 12


class TestStaticStructureFactorDebye:
    def test_zero_k_equals_n(self):
        frame, _ = _octahedron_frame()
        ssf = molrs.StaticStructureFactorDebye([0.0])
        out = ssf.compute(frame)
        k, sk, n = out[0]
        assert n == 7
        assert abs(sk[0] - 7.0) < 1e-10

    def test_linspace_constructor(self):
        ssf = molrs.StaticStructureFactorDebye.linspace(0.5, 5.0, 10)
        assert ssf is not None


class TestPMFTXY:
    def test_two_particles_two_bins(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        f = _make_frame(pts, box_len=10.0)
        nl = _nlist(f, pts, cutoff=1.5)
        out = molrs.PMFTXY(2.0, 2.0, 8, 8).compute(f, nl)
        counts, density, pmf = out[0]
        assert counts.sum() == 2  # one each side, self-query symmetric pair


class TestClusterProperties:
    def test_two_particle_uniform(self):
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        f = _make_frame(pts, box_len=10.0)
        nl = _nlist(f, pts, cutoff=3.0)
        cl_result = molrs.Cluster(1).compute(f, nl)
        out = molrs.ClusterProperties().compute(f, [cl_result])
        d = out[0]
        assert d["sizes"] == [2]
        # Centre of mass at (1, 0, 0).
        assert abs(d["centers_of_mass"][0, 0] - 1.0) < 1e-10
