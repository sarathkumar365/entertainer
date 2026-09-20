import numpy as np

from entertainer.models.encoder import _truncate


def test_truncation_renormalises():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(20, 64)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    out = _truncate(mat, 16)
    assert out.shape == (20, 16)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_truncation_does_not_mutate_the_input():
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(8, 32)).astype(np.float32)
    before = mat.copy()
    _truncate(mat, 8)
    assert np.array_equal(mat, before), "must not normalise through a view of the caller's array"


def test_shorter_than_target_is_passed_through_normalised():
    mat = np.full((3, 4), 2.0, dtype=np.float32)
    out = _truncate(mat, 16)
    assert out.shape == (3, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
