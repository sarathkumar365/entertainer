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

# Skipped when ENTERTAINER_NO_DOTENV is set. Without this, deleting the TMDB
# variables in a test is not enough — .env is re-read and the suite makes
# live API calls with real credentials, which is both slow and a way to leak
# a key into a CI log.
if not os.environ.get("ENTERTAINER_NO_DOTENV"):
    load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    override = os.environ.get("ENTERTAINER_DATA_DIR")
    return Path(override).expanduser() if override else REPO_ROOT / "data"


@dataclass(frozen=True)
class Paths:
    """Where everything lives.

    ``root`` resolves on every access rather than being captured at import.
    Fifteen modules do ``from .config import PATHS``, which binds the *object*,
    so reloading this module rebinds only its own name. Any module missing from
    a test's reload list therefore kept pointing at the real data directory —
    which is how the suite came to overwrite ``data/artifacts/taste.npz``:
    ``models.taste`` was never reloaded, so ``Engine.fit(save=True)`` wrote a
    model fitted on synthetic test titles over the real one.

    Resolving late makes ``ENTERTAINER_DATA_DIR`` authoritative for every
    module at all times, and removes the need for a reload list at all.
    Passing a root explicitly still pins it.
    """

    _root: Path | None = None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else _data_dir()

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


PATHS = Paths()


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
    """Return usable ``(v3 key, v4 bearer)`` credentials.

    Environment files are often edited by hand.  A pasted bearer token with a
    smart quote or other non-ASCII character cannot be sent as an HTTP header;
    treating it as present used to hide a valid API key and fail every web
    request with an opaque 500.  Ignore unusable values here so all TMDB
    callers get the same safe fallback behaviour.
    """
    key = os.environ.get("TMDB_API_KEY") or None
    bearer = os.environ.get("TMDB_BEARER") or None
    return key if key and key.isascii() else None, bearer if bearer and bearer.isascii() else None


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

# Names for every language code TMDB is likely to return. The item card is
# prose fed to a text encoder, and "a Mandarin-language film" carries meaning
# to a multilingual model in a way that "a cmn-language film" does not.
LANGUAGE_NAMES: dict[str, str] = {
    "ab": "Abkhaz", "af": "Afrikaans", "am": "Amharic", "ar": "Arabic",
    "as": "Assamese", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "bs": "Bosnian", "ca": "Catalan",
    "cn": "Cantonese", "cmn": "Mandarin", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "dz": "Dzongkha", "el": "Greek",
    "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian",
    "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "ga": "Irish", "gl": "Galician", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz",
    "la": "Latin", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "nb": "Norwegian",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sh": "Serbo-Croatian", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "xx": "unknown-language", "yi": "Yiddish",
    "zh": "Chinese", "zu": "Zulu",
}


def language_label(code: str | None) -> str:
    """Human-readable language name, falling back to the raw code."""
    if not code:
        return ""
    return LANGUAGE_NAMES.get(code, PRIORITY_LANGUAGES.get(code, code))


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

# Rank the collaborative factors are truncated to before fusion.
#
# iALS spreads variance almost uniformly across its 192 dimensions, but only
# the leading directions are recoverable from text: measured on the real
# catalogue, content predicts the top 16 components with R^2 0.20 and all 192
# with R^2 0.07. The trailing dimensions are idiosyncratic co-watch signal
# that no synopsis contains.
#
# Truncating loses less than it sounds. The top 48 components hold only 38%
# of the raw variance but preserve 98.2% of item-to-item similarity, which is
# the only thing the fused space uses the factors for. Measured:
#
#     rank   imputation R^2   similarity preserved
#       16           0.202                   0.909
#       32           0.158                   0.962
#       48           0.138                   0.982
#       64           0.123                   0.990
#      192           0.075                   1.000
#
# 48 is the knee: near-double the imputation accuracy for the half of the
# catalogue MovieLens never saw, at 1.8% of the similarity structure.
CF_RANK = 48

FUSED_DIM = 192
