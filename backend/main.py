import asyncio
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.parser import (
    LetterboxdMappingError,
    LetterboxdParserError,
    enrich_letterboxd_csv,
)
from src.recommender.hybrid_recommender import hybrid_recommend


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Letterboxd recommender API is running"}


@app.post("/recommend")
async def get_recommendations(
    file: UploadFile = File(...)
):
    temp_path = None

    try:
        filename = file.filename or ""
        if Path(filename).suffix.casefold() != ".csv":
            raise HTTPException(
                status_code=400,
                detail="Please upload your Letterboxd ratings.csv file.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:
            temp_path = temp_file.name
            upload_size = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                upload_size += len(chunk)
                if upload_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The uploaded CSV must be smaller than 10 MB.",
                    )
                temp_file.write(chunk)

        if upload_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV is empty.",
            )

        try:
            enriched_ratings = await run_in_threadpool(
                enrich_letterboxd_csv,
                temp_path,
            )
        except LetterboxdParserError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LetterboxdMappingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        import_summary = dict(
            enriched_ratings.attrs.get("import_summary", {})
        )
        import_summary.setdefault("usable_rows", len(enriched_ratings))
        import_summary.setdefault("total_rows", len(enriched_ratings))

        # build_user_features matches ratings.index against catalogue TMDB IDs.
        # Retain the column too: hybrid_recommend uses it to exclude watched films.
        ratings_for_model = enriched_ratings[["tmdbId", "rating"]].copy()
        ratings_for_model = ratings_for_model.set_index("tmdbId", drop=False)

        async with INFERENCE_SEMAPHORE:
            recommendations = await run_in_threadpool(
                hybrid_recommend,
                ratings_for_model,
            )

        if not isinstance(recommendations, pd.DataFrame):
            raise HTTPException(
                status_code=500,
                detail=(
                    "hybrid_recommend must return its recommendation DataFrame. "
                    "It currently only prints the results."
                ),
            )

        required_columns = {"tmdbId", "movie_title", "xgb_rank"}
        if not required_columns.issubset(recommendations.columns):
            raise HTTPException(
                status_code=500,
                detail="The recommender result must contain tmdbId, movie_title and xgb_rank.",
            )

        import_summary["catalogue_rows"] = recommendations.attrs.get(
            "catalogue_rows",
            import_summary["usable_rows"],
        )

        return {
            "recommendations": json.loads(
                recommendations[["tmdbId", "movie_title", "xgb_rank"]]
                .to_json(orient="records")
            ),
            "import_summary": import_summary,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
