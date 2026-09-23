import pytest

from entertainer import resources
from entertainer.resources import GiB


@pytest.fixture
def ram(monkeypatch):
    def set_ram(gib):
        monkeypatch.setattr(resources, "total_memory", lambda: int(gib * GiB))
    monkeypatch.delenv(resources.MEMORY_FRACTION_ENV, raising=False)
    return set_ram


def test_budget_is_a_share_of_physical_ram(ram):
    ram(16)
    assert resources.budget().bytes == 8 * GiB
    assert resources.budget(0.2).bytes == int(16 * GiB * 0.2)


def test_fraction_comes_from_the_environment(ram, monkeypatch):
    ram(16)
    monkeypatch.setenv(resources.MEMORY_FRACTION_ENV, "0.25")
    assert resources.budget().bytes == 4 * GiB


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_nonsense_fractions_are_refused(bad):
    with pytest.raises(ValueError):
        resources.memory_fraction(bad)


def test_an_eight_gigabyte_laptop_does_not_overlap_cf_with_a_cpu_encoder(ram):
    ram(8)
    assert not resources.can_overlap_cf("cpu")
    assert not resources.can_overlap_cf("mps")


def test_a_server_overlaps_cf_with_the_gpu_encoder(ram):
    ram(32)
    assert resources.can_overlap_cf("cuda")
    assert not resources.can_overlap_cf("cuda", fraction=0.1)


def test_unknown_ram_rules_nothing_out(ram):
    ram(0)
    assert resources.can_overlap_cf("cpu")
    assert resources.duckdb_config() == {}


def test_duckdb_is_capped_at_the_budget(ram):
    ram(8)
    assert resources.duckdb_config() == {"memory_limit": "4096MB"}


def test_a_bad_fraction_does_not_break_database_connections(ram, monkeypatch):
    ram(8)
    monkeypatch.setenv(resources.MEMORY_FRACTION_ENV, "50")
    assert resources.duckdb_config() == {"memory_limit": "4096MB"}
