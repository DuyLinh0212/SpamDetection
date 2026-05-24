import csv
import io
import math
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODEL_PATH = ROOT_DIR / "models" / "xgb_spam_model.pkl"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_extract import SELECTED_FEATURES, extract_feature_dict, rule_based_spam_score  # noqa: E402


TEXT_KEYS = ["comment_text", "Content", "content", "text", "comment", "message", "body"]
TIME_KEYS = ["CreatedAt", "created_at", "comment_time", "time", "timestamp", "date"]
ID_KEYS = ["Id", "id", "comment_id", "cmtid"]
RATING_KEYS = ["rating_star", "rating", "stars", "star", "score"]


app = FastAPI(title="Shopee Spam Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pick_field(record: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        if key in record:
            value = record.get(key)
            return "" if value is None else str(value)
    return default


def pick_float(record: dict[str, Any], candidates: list[str], default: float = 0.0) -> float:
    raw = pick_field(record, candidates)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def safe_number(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: safe_number(value) for key, value in record.items()}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if joblib is not None:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def read_csv_upload(file_bytes: bytes) -> list[dict[str, Any]]:
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must contain a header row.")
    return [dict(row) for row in reader]


def get_model_feature_names(model) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(name) for name in feature_names]

    try:
        booster_names = model.get_booster().feature_names
        if booster_names:
            return [str(name) for name in booster_names]
    except Exception:
        pass

    return SELECTED_FEATURES


def predict_record(model, record: dict[str, Any], row_number: int, threshold: float) -> Optional[dict[str, Any]]:
    text = pick_field(record, TEXT_KEYS)
    if not text.strip():
        return None

    created_at = pick_field(record, TIME_KEYS)
    rating = pick_float(record, RATING_KEYS)
    feature_values = extract_feature_dict(text, created_at, rating)
    model_feature_names = get_model_feature_names(model)
    feature_row = [float(feature_values.get(name, 0.0)) for name in model_feature_names]
    feature_df = pd.DataFrame([feature_row], columns=model_feature_names)

    if hasattr(model, "predict_proba"):
        ml_score = float(model.predict_proba(feature_df)[0][1])
    else:
        ml_score = float(model.predict(feature_df)[0])

    rule_score = rule_based_spam_score(text, created_at, rating)
    spam_score = max(ml_score, rule_score)
    is_spam = int(spam_score >= threshold)

    return {
        "row_number": row_number,
        "id": pick_field(record, ID_KEYS, default=str(row_number)),
        "text": text,
        "created_at": created_at,
        "rating": rating,
        "spam_score": round(spam_score, 6),
        "ml_score": round(ml_score, 6),
        "rule_score": round(rule_score, 6),
        "is_spam": is_spam,
        "label": "Spam" if is_spam else "Normal",
        "features": {name: safe_number(feature_values.get(name, 0.0)) for name in SELECTED_FEATURES},
        "raw": safe_record(record),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_exists = MODEL_PATH.exists()
    return {
        "status": "ok" if model_exists else "missing_model",
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "features": SELECTED_FEATURES,
    }


@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")

    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        records = read_csv_upload(await file.read())
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be encoded as UTF-8 or UTF-8-SIG.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot load model: {exc}") from exc

    results = []
    skipped_rows = 0
    for index, record in enumerate(records, start=1):
        prediction = predict_record(model, record, index, threshold)
        if prediction is None:
            skipped_rows += 1
            continue
        results.append(prediction)

    spam_count = sum(item["is_spam"] for item in results)
    return {
        "filename": file.filename,
        "threshold": threshold,
        "total_rows": len(records),
        "predicted_rows": len(results),
        "skipped_rows": skipped_rows,
        "spam_count": spam_count,
        "normal_count": len(results) - spam_count,
        "results": results,
    }
