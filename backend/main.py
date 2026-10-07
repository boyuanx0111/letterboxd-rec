import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.parser import (
    LetterboxdMappingError,
    LetterboxdParserError,
    enrich_letterboxd_csv,
)
from src.recommender.hybrid_recommender import (
    RecommenderInputError,
    hybrid_recommend,
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        import_summary = dict(enriched_ratings.attrs.get("import_summary", {}))

        try:
            async with INFERENCE_SEMAPHORE:
                recommendations = await run_in_threadpool(
                    hybrid_recommend,
                    enriched_ratings,
                )
        except RecommenderInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        import_summary["catalogue_rows"] = recommendations.attrs.get(
            "catalogue_rows",
            import_summary["usable_rows"],
        )

        return {
            "recommendations": recommendations.to_dict(orient="records"),
            "import_summary": import_summary,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
