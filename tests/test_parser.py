import pytest

from src.parser import LetterboxdParserError, enrich_letterboxd_csv


def test_enriches_existing_tmdb_ids_and_drops_invalid_rows(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text(
        "Name,Year,Rating,TmdbId,TmdbIdType\n"
        "Film,2000,4.5,123,movie\n"
        "TV,2001,5.0,456,tv\n"
        "Unrated,2002,,789,movie\n",
        encoding="utf-8",
    )

    result = enrich_letterboxd_csv(csv_path, cache_path=tmp_path / "cache.json")

    assert result.index.tolist() == [123]
    assert result["rating"].tolist() == [4.5]
    assert result.attrs["import_summary"] == {
        "total_rows": 3,
        "mapped_rows": 3,
        "usable_rows": 1,
        "skipped_rows": 2,
        "mapping_failures": 0,
    }


def test_rejects_csv_without_rating_column(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("Name,TmdbId\nFilm,123\n", encoding="utf-8")

    with pytest.raises(LetterboxdParserError, match="missing the Rating column"):
        enrich_letterboxd_csv(csv_path, cache_path=tmp_path / "cache.json")


def test_rejects_unmapped_rows_without_letterboxd_uri(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("Name,Year,Rating\nFilm,2000,4\n", encoding="utf-8")

    with pytest.raises(LetterboxdParserError, match="missing: Letterboxd URI"):
        enrich_letterboxd_csv(csv_path, cache_path=tmp_path / "cache.json")
