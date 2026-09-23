"""Where the build has got to, and what to do next.

The preflight checks live here too, kept apart from the expensive stages so
that a full build can fail in the first second rather than the third hour.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import IMDB_FILES, PATHS, has_tmdb
from .errors import EntertainerError, MissingCredentials

MIN_FREE_BYTES = 10 * 1024**3

#: Below this share of the catalogue embedded into the item space, the gap is
#: worth warning about: anything outside the space cannot be recommended at
#: all, so a silent 3% shortfall is 2,000 titles that simply never appear.
STALE_COVERAGE = 0.995

#: Verdicts needed before recommendations mean anything, and before the audit
#: can say anything honest. Mirrors taste.fit and prequential respectively.
MIN_FOR_RECS = 3
MIN_FOR_AUDIT = 8


def preflight(require_tmdb: bool = True) -> dict[str, int | bool | str]:
    """Fail early when a local full build cannot complete safely."""
    PATHS.ensure()
    free = shutil.disk_usage(PATHS.root).free
    if free < MIN_FREE_BYTES:
        raise EntertainerError(
            f"{free / 1024**3:.1f} GiB free at {PATHS.root}; the local build needs "
            "at least 10 GiB. Raw IMDb downloads can be deleted after a successful build."
        )
    if require_tmdb and not has_tmdb():
        raise MissingCredentials(
            "no usable TMDB credential: set an ASCII TMDB_API_KEY or TMDB_BEARER in .env"
        )
    return {"data_dir": str(PATHS.root), "free_bytes": free, "tmdb": has_tmdb()}


@dataclass
class Step:
    """What to run next, and why.

    ``note`` is the aside the CLI prints in dim text. It is separate from the
    command so that a caller which is not a terminal can use one without the
    other — the previous version returned a single string with rich markup
    embedded in it, which made this function unusable anywhere else.
    """

    command: str
    note: str = ""


def catalogue_exists() -> bool:
    return PATHS.catalog_db.exists()


def next_step(present: dict[str, bool], n_ratings: int) -> Step:
    """The first unmet dependency, in build order.

    A pipeline with eight stages and a three-hour critical path needs to be
    able to say where it got to.
    """
    if not present["catalog"]:
        return Step("ent setup")
    if not present["content_embeddings"]:
        return Step("ent data embed")
    if not present["cf_factors"]:
        return Step("ent data cf")
    if not present["fused_space"]:
        return Step("ent data fuse")
    if n_ratings < MIN_FOR_RECS:
        return Step("ent onboard", "or: ent bulk seed.example.txt")
    if n_ratings < MIN_FOR_AUDIT:
        return Step("ent recs", "a few more verdicts and `ent audit` will work too")
    return Step("ent recs")


def coverage_is_stale(coverage: dict[str, int]) -> float | None:
    """The embedded share, when it is low enough to be worth reporting.

    Returns None when coverage is fine or cannot be computed, so the caller
    does not repeat the threshold comparison.
    """
    if not coverage.get("catalog") :
        return None
    share = coverage["fused"] / coverage["catalog"]
    return share if share < STALE_COVERAGE else None


#: The eight stages of a full build, in dependency order, with the label the
#: Reporter records them under.
STAGES = (
    ("sources", "downloading source data"),
    ("catalogue", "building catalogue"),
    ("tmdb", "enriching from TMDB"),
    ("prune", "pruning by language"),
    ("embeddings", "encoding item text"),
    ("cf", "factorising MovieLens"),
    ("fusion", "fusing item space"),
    ("prior", "learning the population prior"),
)


@dataclass
class BuildSettings:
    """Every knob a full build turns.

    These were literals scattered through the `ent setup` command body, which
    made them invisible: nothing said that the fused space is 192-dimensional
    because the collaborative factorisation produces 192 factors, or that the
    keyword pass targets the top 150,000 titles by vote count rather than a
    batch size. Naming them together is the point — several have to agree
    with each other.
    """

    skip_enrich: bool = False
    enrich_limit: int = 0
    min_votes: int = 50
    floor_scale: float = 1.0

    #: TMDB allows a lot of concurrency; this is the rate the API tolerates
    #: rather than anything about this machine.
    concurrency: int = 40
    #: A coverage target, not a batch size: the most-voted N titles get
    #: keywords, and the pass stops when they have them.
    keyword_target: int = 150_000

    #: None sizes the batch to the device (see encoder.auto_batch_size).
    encode_batch: int | None = None

    #: Must match `fusion_dim`. The fused space is a rotation of the
    #: collaborative and content blocks, so asking for more fused dimensions
    #: than there are factors would pad it with noise.
    cf_factors: int = 192
    cf_iterations: int = 20
    #: Users withheld from the factorisation and reserved for the offline
    #: benchmark. See evaluation.integrity.
    cf_holdout: int = 2_000
    cf_signal: str = "watched"

    fusion_dim: int = 192

    prior_max_users: int = 20_000
    prior_shrinkage: float = 0.15

    #: Factorise MovieLens in a separate process while the catalogue, TMDB
    #: and embedding stages run. It needs only ratings.csv, so it has no
    #: reason to wait for them.
    parallel: bool = True
    #: Rerun the catalogue and factorisation even when their inputs are
    #: unchanged since the last successful run.
    force: bool = False

    def __post_init__(self) -> None:
        if self.fusion_dim > self.cf_factors:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) exceeds cf_factors ({self.cf_factors}); "
                "the extra dimensions would be noise"
            )


# --- input fingerprints ------------------------------------------------------
#
# A stage whose inputs have not changed since it last succeeded produces the
# same output, so a rerun can skip it. The fingerprint is what the stage
# reads plus the settings that shape it; bump the version constant whenever
# the stage's own code changes what it would produce from the same inputs.

CATALOGUE_VERSION = 1
CF_VERSION = 1
CATALOGUE_INPUTS_KEY = "catalogue_inputs"
#: The published build this catalogue was pulled from, if it was pulled.
PUBLISHED_BUILD_KEY = "catalogue_build"


def _file_stamp(path: Path) -> str:
    if not path.exists():
        return "missing"
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def catalogue_inputs(settings: BuildSettings) -> str:
    parts = [f"version={CATALOGUE_VERSION}"]
    for name in IMDB_FILES:
        path = PATHS.raw / "imdb" / name
        # The server's ETag names the file's version exactly; a size and mtime
        # is the fallback for a copy downloaded before validators were kept.
        validator = path.with_name(path.name + ".validator")
        version = validator.read_text().strip() if validator.exists() else ""
        parts.append(f"{name}={version or _file_stamp(path)}")
    parts.append(f"links={_file_stamp(PATHS.raw / 'ml-32m' / 'links.csv')}")
    # The prune deletes rows in place, so a catalogue built at one floor
    # cannot be reused at a lower one.
    parts.append(f"min_votes={settings.min_votes}")
    parts.append(f"floor_scale={settings.floor_scale}")
    return _digest(parts)


def cf_inputs(settings: BuildSettings) -> str:
    return _digest([
        f"version={CF_VERSION}",
        f"ratings={_file_stamp(PATHS.raw / 'ml-32m' / 'ratings.csv')}",
        f"factors={settings.cf_factors}",
        f"iterations={settings.cf_iterations}",
        f"holdout={settings.cf_holdout}",
        f"signal={settings.cf_signal}",
    ])


def cf_inputs_path() -> Path:
    return PATHS.embeddings / "cf_inputs.sha256"


def cf_is_current(settings: BuildSettings) -> bool:
    from .evaluation import integrity

    stamp = cf_inputs_path()
    return (
        stamp.exists()
        and stamp.read_text().strip() == cf_inputs(settings)
        and (PATHS.embeddings / "cf_factors.npy").exists()
        # The prior and the benchmark read the holdout the factorisation chose.
        and (settings.cf_holdout <= 0 or integrity.holdout_path().exists())
    )


def record_cf_inputs(fingerprint: str) -> None:
    PATHS.ensure()
    stamp = cf_inputs_path()
    tmp = stamp.with_name(f".{stamp.name}.tmp")
    tmp.write_text(fingerprint)
    tmp.replace(stamp)
