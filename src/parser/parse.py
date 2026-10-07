"""Parse a Letterboxd ratings export and add external movie identifiers.

The mapping approach is adapted from Tetrax-10's
letterboxd-csv-imdb-tmdb-mapper and remains covered by its MIT licence. See
``LETTERBOXD_MAPPER_LICENSE`` in this directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd


MAX_CONCURRENT_REQUESTS = 10
DEFAULT_TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "letterboxd" / "cache.json"
)

_ALLOWED_HOSTS = {"boxd.it", "letterboxd.com", "www.letterboxd.com"}
_TMDB_PATTERN = re.compile(r"/(movie|tv)/(\d+)(?:[/?#]|$)", re.IGNORECASE)
_BOXD_SLUG_PATTERN = re.compile(r"boxd\.it/([a-zA-Z0-9]+)", re.IGNORECASE)
_CACHE_SEPARATOR = "|:|"
_CACHE_LOCK = threading.Lock()
_FETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_REQUESTS,
    thread_name_prefix="letterboxd-mapper",
)
logger = logging.getLogger(__name__)


class LetterboxdParserError(ValueError):
    """Raised when an uploaded file is not a usable Letterboxd ratings CSV."""


class LetterboxdMappingError(RuntimeError):
    """Raised when Letterboxd pages cannot be mapped to TMDB IDs."""

# Small data class for the identifiers associated with a title.
@dataclass(frozen=True)
class ExternalIds:
    """Store the TMDB identifier and media type extracted for one title."""

    tmdb_type: str = ""
    tmdb_id: int | None = None


class _ExternalLinkParser(HTMLParser):
    """Collect the TMDB link from Letterboxd anchor tags."""

    def __init__(self) -> None:
        """Initialize the HTML parser and its empty link results."""

        super().__init__(convert_charrefs=True)
        self.tmdb_href = ""

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Record a link when an anchor identifies TMDB."""

        if tag.casefold() != "a":
            return

        attributes = dict(attrs)
        action = (attributes.get("data-track-action") or "").casefold()
        href = attributes.get("href") or ""

        if action == "tmdb":
            self.tmdb_href = href


def _normalise_url(value: Any) -> str:
    """Validate and return an HTTPS URL hosted by Letterboxd."""

    url = str(value or "").strip()
    parsed = urlparse(url)

    if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in _ALLOWED_HOSTS:
        raise LetterboxdParserError(
            "Every unmapped row must contain an HTTPS Letterboxd URI."
        )

    return url


class _LetterboxdRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Allow redirects only when their destination is still Letterboxd."""

        redirect_url = urljoin(req.full_url, newurl)
        try:
            _normalise_url(redirect_url)
        except LetterboxdParserError as exc:
            raise URLError("Letterboxd redirected to an unsupported host.") from exc
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            redirect_url,
        )


def _extract_external_ids(html: str) -> ExternalIds:
    """Parse Letterboxd HTML and return its linked TMDB ID and media type."""

    parser = _ExternalLinkParser()
    parser.feed(html)

    tmdb_match = _TMDB_PATTERN.search(parser.tmdb_href)

    if not tmdb_match:
        raise LetterboxdMappingError("The Letterboxd page did not contain a TMDB link.")

    return ExternalIds(
        tmdb_type=tmdb_match.group(1).casefold(),
        tmdb_id=int(tmdb_match.group(2)),
    )


def _fetch_external_ids(url: str, timeout: float) -> ExternalIds:
    """Fetch a Letterboxd page safely and extract its external identifiers."""

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    try:
        opener = build_opener(_LetterboxdRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            try:
                _normalise_url(response.geturl())
            except LetterboxdParserError as exc:
                raise LetterboxdMappingError(
                    "Letterboxd redirected to an unsupported host."
                ) from exc

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_RESPONSE_BYTES:
                raise LetterboxdMappingError(
                    "The Letterboxd page was larger than expected."
                )
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise LetterboxdMappingError(
                    "The Letterboxd page was larger than expected."
                )
            html = body.decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise LetterboxdMappingError(f"Could not fetch {url}: {exc}") from exc

    return _extract_external_ids(html)


def _normalise_year(value: Any) -> str:
    """Format a year consistently so it matches existing mapper cache keys."""

    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _cache_identity(row: pd.Series, url: str) -> tuple[str, str]:
    """Build the exact cache key and legacy name/year lookup prefix."""

    name = str(row.get("Name", "")).strip()
    year = _normalise_year(row.get("Year"))
    slug_match = _BOXD_SLUG_PATTERN.search(url)
    slug = slug_match.group(1) if slug_match else url
    prefix = f"{name}{_CACHE_SEPARATOR}{year}{_CACHE_SEPARATOR}"
    return f"{prefix}{slug}", prefix


def _decode_cache_value(value: Any) -> ExternalIds | None:
    """Convert one stored cache entry into IDs, or return None if invalid."""

    if isinstance(value, str):
        parts = value.split("|")
        tmdb_type = parts[0] if parts else ""
        tmdb_id = parts[1] if len(parts) > 1 else ""
    elif isinstance(value, dict):
        tmdb_type = value.get("tmdb_type", "")
        tmdb_id = value.get("tmdb_id")
    else:
        return None

    try:
        parsed_tmdb_id = int(tmdb_id)
    except (TypeError, ValueError):
        return None

    return ExternalIds(
        tmdb_type=str(tmdb_type).casefold(),
        tmdb_id=parsed_tmdb_id,
    )


def _load_cache(cache_path: Path) -> dict[str, Any]:
    """Read a cache dictionary, returning empty when it is absent or invalid."""

    if not cache_path.exists():
        return {}

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable Letterboxd cache %s: %s", cache_path, exc)
        return {}

    return cache if isinstance(cache, dict) else {}


def _find_cached_ids(
    cache: dict[str, Any],
    cache_key: str,
    cache_prefix: str,
) -> ExternalIds | None:
    """Look up an exact cache key, then a unique legacy name/year match."""

    cached = _decode_cache_value(cache.get(cache_key))
    if cached:
        return cached

    legacy_matches = [
        cached
        for key, value in cache.items()
        if key.startswith(cache_prefix)
        if (cached := _decode_cache_value(value)) is not None
    ]
    return legacy_matches[0] if len(legacy_matches) == 1 else None


def _save_cache(cache_path: Path, updates: dict[str, ExternalIds]) -> None:
    """Merge new IDs into the JSON cache and replace it atomically."""

    if not updates:
        return

    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
        cache.update(
            {
                key: f"{ids.tmdb_type}|{ids.tmdb_id}"
                for key, ids in updates.items()
            }
        )
        temporary_path = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f"{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                json.dump(cache, temporary_file, indent=2, ensure_ascii=False)
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, cache_path)
        except OSError as exc:
            logger.warning("Could not update Letterboxd cache %s: %s", cache_path, exc)
        finally:
            if temporary_path:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary cache file %s: %s",
                        temporary_path,
                        exc,
                    )


def _normalise_existing_columns(ratings: pd.DataFrame) -> pd.DataFrame:
    """Trim headers and normalize mapper output columns for the recommender."""

    ratings = ratings.copy()
    ratings.columns = [str(column).strip() for column in ratings.columns]

    aliases = {
        "Rating": "rating",
        "TmdbId": "tmdbId",
        "TmdbIdType": "tmdbType",
    }
    for source, destination in aliases.items():
        if source in ratings.columns and destination not in ratings.columns:
            ratings = ratings.rename(columns={source: destination})

    return ratings


def enrich_letterboxd_csv(
    csv_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Return valid movie ratings enriched with TMDB identifiers.

    Raw Letterboxd rows are mapped through their ``Letterboxd URI`` (or the
    mapper's ``URL`` column). Previously enriched mapper CSVs are accepted too.
    The result contains only the normalized ``tmdbId`` and ``rating`` columns,
    indexed by ``tmdbId``, so it can be passed directly to the recommender.
    Import counts are stored in ``DataFrame.attrs`` for the API response.
    """

    try:
        ratings = pd.read_csv(csv_path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise LetterboxdParserError(f"Could not read the uploaded CSV: {exc}") from exc

    if ratings.empty:
        raise LetterboxdParserError("The uploaded CSV contains no ratings.")

    ratings = _normalise_existing_columns(ratings)
    if "rating" not in ratings.columns:
        raise LetterboxdParserError("The uploaded CSV is missing the Rating column.")

    if "tmdbId" not in ratings.columns:
        ratings["tmdbId"] = pd.NA
    if "tmdbType" not in ratings.columns:
        ratings["tmdbType"] = pd.NA

    ratings["tmdbId"] = pd.to_numeric(ratings["tmdbId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    rows_to_map = ratings.index[ratings["tmdbId"].isna()].tolist()

    url_column = next(
        (column for column in ("Letterboxd URI", "URL") if column in ratings.columns),
        None,
    )
    if rows_to_map:
        required_columns = {"Name", "Year"}
        missing_columns = required_columns.difference(ratings.columns)
        if url_column is None:
            missing_columns.add("Letterboxd URI")
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise LetterboxdParserError(f"The uploaded CSV is missing: {missing}.")

        missing_url_rows = [
            index
            for index in rows_to_map
            if pd.isna(ratings.at[index, url_column])
            or not str(ratings.at[index, url_column]).strip()
        ]
        if missing_url_rows:
            raise LetterboxdParserError(
                f"{len(missing_url_rows)} row(s) are missing a Letterboxd URI."
            )

    resolved_cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    with _CACHE_LOCK:
        cache = _load_cache(resolved_cache_path)

    pending: dict[str, tuple[str, list[Any]]] = {}
    mapping_errors: list[str] = []

    for index in rows_to_map:
        row = ratings.loc[index]
        try:
            url = _normalise_url(row[url_column])
        except LetterboxdParserError as exc:
            mapping_errors.append(str(exc))
            continue

        cache_key, cache_prefix = _cache_identity(row, url)
        cached_ids = _find_cached_ids(cache, cache_key, cache_prefix)
        if cached_ids:
            ratings.at[index, "tmdbId"] = cached_ids.tmdb_id
            ratings.at[index, "tmdbType"] = cached_ids.tmdb_type
            continue

        if cache_key not in pending:
            pending[cache_key] = (url, [])
        pending[cache_key][1].append(index)

    cache_updates: dict[str, ExternalIds] = {}
    if pending:
        worker_count = min(
            max(1, int(max_workers)),
            MAX_CONCURRENT_REQUESTS,
            len(pending),
        )
        request_limit = threading.BoundedSemaphore(worker_count)

        def fetch_with_limit(url: str) -> ExternalIds:
            """Fetch one URL while respecting this upload's worker limit."""

            with request_limit:
                return _fetch_external_ids(url, timeout)

        futures = {
            _FETCH_EXECUTOR.submit(fetch_with_limit, url): cache_key
            for cache_key, (url, _) in pending.items()
        }
        for future in as_completed(futures):
            cache_key = futures[future]
            try:
                external_ids = future.result()
            except LetterboxdMappingError as exc:
                mapping_errors.append(str(exc))
                continue

            cache_updates[cache_key] = external_ids
            for index in pending[cache_key][1]:
                ratings.at[index, "tmdbId"] = external_ids.tmdb_id
                ratings.at[index, "tmdbType"] = external_ids.tmdb_type

    _save_cache(resolved_cache_path, cache_updates)

    total_rows = len(ratings)
    ratings["tmdbType"] = (
        ratings["tmdbType"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .replace({"movie": "movie", "tv series": "tv"})
    )

    mapped_rows = int(ratings["tmdbId"].notna().sum())
    valid_type = ratings["tmdbType"].isna() | ratings["tmdbType"].eq("movie")
    usable = ratings[
        ratings["tmdbId"].notna()
        & ratings["tmdbId"].gt(0)
        & ratings["rating"].gt(0)
        & ratings["rating"].le(5)
        & valid_type
    ].copy()

    if usable.empty:
        if pending and mapping_errors:
            raise LetterboxdMappingError(
                "No films could be mapped. Letterboxd may be unavailable; try again shortly."
            )
        if mapping_errors:
            raise LetterboxdParserError(mapping_errors[0])
        raise LetterboxdParserError(
            "The CSV contains no rated movies with usable TMDB IDs."
        )

    usable["tmdbId"] = usable["tmdbId"].astype("int64")
    usable["rating"] = usable["rating"].astype("float64")
    usable = usable.drop_duplicates(subset="tmdbId", keep="last")
    summary = {
        "total_rows": total_rows,
        "mapped_rows": mapped_rows,
        "usable_rows": len(usable),
        "skipped_rows": total_rows - len(usable),
        "mapping_failures": len(mapping_errors),
    }
    usable = usable[["tmdbId", "rating"]].set_index("tmdbId", drop=False)
    usable.attrs.update(
        {
            **summary,
            "import_summary": summary,
        }
    )
    return usable


__all__ = [
    "LetterboxdMappingError",
    "LetterboxdParserError",
    "enrich_letterboxd_csv",
]
