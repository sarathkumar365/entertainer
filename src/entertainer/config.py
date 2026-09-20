"""Runtime configuration.

Everything the engine writes lives under a single data directory so that the
whole state (catalogue, embeddings, your taste profile) can be backed up or
thrown away as one unit. Nothing here is committed to git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    override = os.environ.get("ENTERTAINER_DATA_DIR")
    return Path(override).expanduser() if override else REPO_ROOT / "data"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def catalog_db(self) -> Path:
        return self.root / "entertainer.duckdb"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def artifacts(self) -> Path:
        """Fitted model state: ALS factors, projections, user posterior."""
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> Paths:
        for p in (self.raw, self.interim, self.embeddings, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths(_data_dir())


# --- Data sources -----------------------------------------------------------

IMDB_BASE = "https://datasets.imdbws.com"
IMDB_FILES = (
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.akas.tsv.gz",
    "title.crew.tsv.gz",
    "title.principals.tsv.gz",
    "name.basics.tsv.gz",
)
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

TMDB_API_BASE = "https://api.themoviedb.org/3"


def tmdb_credentials() -> tuple[str | None, str | None]:
    """Return (v3 api key, v4 bearer token). Either is enough; bearer wins."""
    return os.environ.get("TMDB_API_KEY") or None, os.environ.get("TMDB_BEARER") or None


def has_tmdb() -> bool:
    key, bearer = tmdb_credentials()
    return bool(key or bearer)


# --- Catalogue scope --------------------------------------------------------

# Title types worth recommending. IMDb carries a long tail of episodes, shorts
# and video games that would only ever be noise here.
KEPT_TITLE_TYPES = frozenset({"movie", "tvMovie", "tvSeries", "tvMiniSeries"})

# Languages the catalogue must cover well, in priority order. Used to set
# per-language vote floors so that a well-regarded Malayalam film is not
# filtered out by thresholds calibrated on Hollywood vote counts.
PRIORITY_LANGUAGES: dict[str, str] = {
    "ta": "Tamil",
    "ml": "Malayalam",
    "te": "Telugu",
    "kn": "Kannada",
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "pt": "Portuguese",
    "zh": "Chinese",
    "cn": "Chinese",
    "ru": "Russian",
    "fa": "Persian",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "th": "Thai",
    "id": "Indonesian",
    "bn": "Bengali",
    "mr": "Marathi",
}

# IMDb regions that stand in for a language when no language tag is available.
REGION_TO_LANGUAGE = {
    "IN": None,  # ambiguous, resolved via akas language column
    "KR": "ko",
    "JP": "ja",
    "FR": "fr",
    "ES": "es",
    "DE": "de",
    "IT": "it",
    "SE": "sv",
    "DK": "da",
    "NO": "no",
    "FI": "fi",
}

# Minimum IMDb votes for a title to enter the catalogue. Small-industry
# languages get a far lower floor: 500 votes on a Malayalam film is roughly the
# cultural footprint of 50,000 votes on an American one.
VOTE_FLOOR_DEFAULT = 2000
VOTE_FLOOR_BY_LANGUAGE: dict[str, int] = {
    "ta": 200,
    "ml": 120,
    "te": 200,
    "kn": 100,
    "bn": 120,
    "mr": 120,
    "hi": 400,
    "ko": 300,
    "ja": 400,
    "fa": 150,
    "th": 200,
    "id": 200,
    "tr": 200,
    "sv": 300,
    "da": 300,
    "no": 300,
    "fi": 200,
    "pl": 300,
    "nl": 300,
    "pt": 300,
    "es": 500,
    "fr": 500,
    "it": 500,
    "de": 500,
    "ru": 400,
    "zh": 400,
    "cn": 400,
    "en": 2000,
}

MIN_YEAR = 1930


# --- Model defaults ---------------------------------------------------------

ENCODER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ENCODER_FALLBACK = "intfloat/multilingual-e5-base"
ENCODER_DIM_TARGET = 256  # Matryoshka truncation of the raw encoder output.

CF_FACTORS = 192
FUSED_DIM = 192
